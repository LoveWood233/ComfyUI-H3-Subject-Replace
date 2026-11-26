"""Endpoint guides for each H3 sampler and optional pixel pins for the exported clip."""

import torch

import comfy.utils
from comfy_api.latest import io


def _endpoints(start, length, first_frame, last_frame):
    anchors = []
    for name, index, image in (
        ("first_frame", start, first_frame),
        ("last_frame", start + length - 1, last_frame),
    ):
        if image is None:
            continue
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[0] != 1
            or min(image.shape[1:3]) <= 0
            or image.shape[-1] not in (3, 4)
            or not image.is_floating_point()
        ):
            raise ValueError(f"{name} must be one RGB or RGBA IMAGE [1, H, W, C].")
        # The native H3 guide and the rendered MP4 both use RGB.
        rgb = image[..., :3]
        if anchors and index == anchors[0][0]:
            if not torch.equal(rgb, anchors[0][1]):
                raise ValueError("A one-frame export cannot have two different endpoint images.")
            continue
        anchors.append((index, rgb))
    return anchors


def _frame_count(tokens):
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN

    cycles, remainder = divmod(tokens, len(FRAME_PER_TOKEN))
    return cycles * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:remainder])


def _video_stream(latent):
    samples = latent.get("samples")
    streams = getattr(samples, "tensors", ())
    if not getattr(samples, "is_nested", False) or len(streams) != 2:
        raise ValueError("Expected a MiniMax H3 audio/video latent from the relevant sampler stage.")
    video = streams[0]
    if video.ndim != 5 or video.shape[1] != 24 or min(video.shape) <= 0:
        raise ValueError("Expected a MiniMax H3 video stream with shape [B, 24, T, H, W].")
    return video


class H3ClipEndpointGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ClipEndpointGuide",
            display_name="H3 Clip Endpoint Guide",
            category="YCNodes-MiniMax-H3/conditioning",
            description="Anchor the first or last exported frame on the current H3 sampling canvas.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Int.Input(
                    "visible_frames",
                    default=124,
                    min=1,
                    max=9999,
                    tooltip="Number of output frames retained after trimming.",
                ),
                io.Int.Input(
                    "trim_start",
                    default=0,
                    min=0,
                    max=9999,
                    tooltip="Number of decoded frames removed from the start before export.",
                ),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, vae, visible_frames, trim_start, first_frame=None, last_frame=None):
        if first_frame is None and last_frame is None:
            return io.NodeOutput(positive)
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
        except ImportError as exc:
            raise RuntimeError(
                "Update ComfyUI to a version with MiniMaxH3AddGuide to anchor frames inside the exported clip."
            ) from exc

        if (
            type(visible_frames) is not int
            or visible_frames < 1
            or type(trim_start) is not int
            or trim_start < 0
        ):
            raise ValueError(
                "visible_frames must be a positive integer and trim_start a nonnegative integer."
            )
        video = _video_stream(latent)
        available = _frame_count(video.shape[2])
        if trim_start + visible_frames > available:
            raise ValueError(
                f"Export window [{trim_start}, {trim_start + visible_frames}) exceeds {available} decoded frames."
            )
        anchors = _endpoints(trim_start, visible_frames, first_frame, last_frame)
        if not positive:
            raise ValueError("Expected positive conditioning from the reference-to-video node.")
        for _, metadata in positive:
            for existing in metadata.get("minimax_keyframes", []):
                previous = existing.get("latent")
                if previous is None:
                    continue  # An audio-only guide may share a frame with a visual anchor.
                if previous.ndim != 5 or previous.shape[-2:] != video.shape[-2:]:
                    raise ValueError("An existing visual guide belongs to a different H3 sampling canvas.")
                start = existing["resolved_frame_index"]
                stop = start + _frame_count(previous.shape[2])
                if any(start <= index < stop for index, _ in anchors):
                    raise ValueError(
                        f"An endpoint overlaps the existing visual guide at frames [{start}, {stop})."
                    )

        # Encode each endpoint once on this stage's actual latent canvas. Preserve
        # different pre-existing keyframe lists if the conditioning has multiple entries.
        native = [positive[0]]
        for index, image in anchors:
            native = MiniMaxH3AddGuide.execute(
                positive=native,
                latent=latent,
                vae=vae,
                frame_idx=index,
                image=image,
            ).result[0]
        added = native[0][1]["minimax_keyframes"][-len(anchors) :]
        return io.NodeOutput(
            [
                [
                    embedding,
                    {**metadata, "minimax_keyframes": [*metadata.get("minimax_keyframes", []), *added]},
                ]
                for embedding, metadata in positive
            ]
        )


class H3PinClipEndpoints(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3PinClipEndpoints",
            display_name="H3 Pin Exported Clip Endpoints",
            category="YCNodes-MiniMax-H3/image",
            description="Set the first and/or last RGB frame after the final trim and before encoding.",
            inputs=[
                io.Image.Input("images"),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, first_frame=None, last_frame=None):
        if first_frame is None and last_frame is None:
            return io.NodeOutput(images)
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[-1] != 3
            or min(images.shape) <= 0
            or not images.is_floating_point()
        ):
            raise ValueError("Expected a nonempty decoded RGB frame batch [frames, H, W, 3].")
        count, height, width, _ = images.shape
        anchors = _endpoints(0, count, first_frame, last_frame)
        # Comfy caches and can branch the input batch. Copy it once, only when pins are in use.
        output = images.clone()
        for index, rgb in anchors:
            if rgb.shape[1:3] != (height, width):
                rgb = comfy.utils.common_upscale(
                    rgb.movedim(-1, 1), width, height, "lanczos", "center"
                ).movedim(1, -1)
            output[index].copy_(rgb[0])
        return io.NodeOutput(output)


NODE_CLASS_MAPPINGS = {
    "H3ClipEndpointGuide": H3ClipEndpointGuide,
    "H3PinClipEndpoints": H3PinClipEndpoints,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ClipEndpointGuide": "H3 Clip Endpoint Guide",
    "H3PinClipEndpoints": "H3 Pin Exported Clip Endpoints",
}
