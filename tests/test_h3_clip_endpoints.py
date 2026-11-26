"""Test endpoint nodes with a real ComfyUI CPU runtime and small tensor fixtures.

Run with ComfyUI's Python environment (no model weights are required)::

    COMFYUI_PATH=/path/to/ComfyUI python tests/test_h3_clip_endpoints.py --cpu -v

Only VAE encoding is doubled. Native H3 guide handling, Comfy's schemas, prompt
validation, execution, tensor resizing, and optional FFmpeg export are real.
"""

import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def configure_comfy_arguments():
    """Let Comfy parse its CPU flag without consuming unittest's arguments."""
    comfy_path = os.environ.get("COMFYUI_PATH")
    if comfy_path:
        sys.path.insert(0, str(Path(comfy_path).resolve()))

    import comfy.options

    original_argv = sys.argv
    original_parsing = comfy.options.args_parsing
    comfy_arguments = ["--cpu"] if "--cpu" in original_argv else []
    try:
        sys.argv = [original_argv[0], *comfy_arguments]
        comfy.options.enable_args_parsing()
        import comfy.cli_args

        if "--cpu" in comfy_arguments:
            comfy.cli_args.args.cpu = True
    finally:
        sys.argv = original_argv
        comfy.options.enable_args_parsing(original_parsing)


configure_comfy_arguments()

import numpy as np
import torch

import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, temporal_shape
import execution
import nodes


NODE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ComfyUI-Installs/custom_nodes/ComfyUI-YCNodes-MiniMax-H3/py/h3_clip_endpoints.py"
)
spec = importlib.util.spec_from_file_location("tested_h3_clip_endpoints", NODE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Guide = module.H3ClipEndpointGuide
Pin = module.H3PinClipEndpoints


class RecordingVAEDouble:
    """Encodes no model: records real resized tensors and returns a small tensor."""

    def __init__(self):
        self.calls = []

    def encode(self, images):
        self.calls.append(images.clone())
        return (
            torch.ones((1, 24, 1, images.shape[1] // 16, images.shape[2] // 16))
            * images.mean()
        )


def image(h=32, w=48, offset=0.0, channels=3):
    base = torch.linspace(0.0, 0.8, h * w * 3).reshape(1, h, w, 3)
    base = (base + offset).clamp(0, 1)
    return (
        torch.cat([base, torch.ones((1, h, w, 1)) * 0.3], dim=-1)
        if channels == 4
        else base
    )


def latent(length=120, w=64, h=48):
    return EmptyMiniMaxH3LatentAV.execute(w, h, length).result[0]


def positive(metadata=None):
    return [[torch.zeros((1, 2, 3)), metadata or {"marker": "reference conditioning"}]]


def keyframes(result):
    return result.result[0][0][1]["minimax_keyframes"]


class EndpointTests(unittest.TestCase):

    def test_real_comfy_schemas_and_registration(self):
        for cls in (Guide, Pin):
            self.assertTrue(issubclass(cls, io.ComfyNode))
            schema = cls.GET_SCHEMA()
            info = cls.GET_NODE_INFO_V1()
            self.assertEqual(info["name"], schema.node_id)
            self.assertIn(schema.node_id, module.NODE_CLASS_MAPPINGS)
            self.assertEqual(len(schema.outputs), 1)

    def test_native_120_frame_request_rounds_to_124(self):
        self.assertEqual(temporal_shape(120)[0], 124)
        self.assertEqual(
            module._frame_count(latent()["samples"].tensors[0].shape[2]), 124
        )

    def test_guide_disconnected_is_identity_without_vae_work(self):
        cond = positive()
        vae = RecordingVAEDouble()
        result = Guide.execute(cond, latent(), vae, 120, 0).result[0]
        self.assertIs(result, cond)
        self.assertEqual(vae.calls, [])

    def test_guide_default_single_and_dual(self):
        first, last = (image(), image(offset=0.1))
        for f, l, expected in [
            (first, None, [0]),
            (None, last, [119]),
            (first, last, [0, 119]),
        ]:
            with self.subTest(first=f is not None, last=l is not None):
                cond = positive()
                vae = RecordingVAEDouble()
                out = Guide.execute(cond, latent(), vae, 120, 0, f, l)
                self.assertEqual(
                    [x["resolved_frame_index"] for x in keyframes(out)], expected
                )
                self.assertEqual(len(vae.calls), len(expected))
                self.assertNotIn("minimax_keyframes", cond[0][1])
                for call in vae.calls:
                    self.assertEqual(tuple(call.shape), (1, 48, 64, 3))

    def test_guide_five_frame_trim_and_bounds(self):
        v = RecordingVAEDouble()
        f = image()
        l = image(offset=0.1)
        out = Guide.execute(positive(), latent(), v, 119, 5, f, l)
        self.assertEqual([x["resolved_frame_index"] for x in keyframes(out)], [5, 123])
        with self.assertRaisesRegex(ValueError, "exceeds 124"):
            Guide.execute(positive(), latent(), v, 120, 5, f, l)
        out = Guide.execute(positive(), latent(141), v, 120, 5, f, l)
        self.assertEqual([x["resolved_frame_index"] for x in keyframes(out)], [5, 124])

    def test_each_sampler_canvas_is_used(self):
        f = image(29, 73, channels=4)
        for w, h in [(32, 32), (64, 48), (96, 64)]:
            vae = RecordingVAEDouble()
            out = Guide.execute(positive(), latent(w=w, h=h), vae, 120, 0, f, f)
            self.assertEqual([tuple(x.shape) for x in vae.calls], [(1, h, w, 3)] * 2)
            self.assertEqual(
                [tuple(x["latent"].shape[-2:]) for x in keyframes(out)],
                [(h // 16, w // 16)] * 2,
            )

    def test_existing_references_audio_and_nonoverlap_preserved_per_entry(self):
        old1 = {"resolved_frame_index": 40, "latent": torch.ones(1, 24, 1, 3, 4)}
        old2 = {"resolved_frame_index": 60, "audio_latent": torch.ones(1, 32, 2, 1)}
        ref = [
            {
                "kind": "image",
                "latent_h": 2,
                "latent_w": 2,
                "latent": torch.ones(1, 24, 1, 2, 2),
            },
            {
                "kind": "audio",
                "ref_audio_t": 3,
                "audio_latent": torch.ones(1, 32, 2, 3),
            },
        ]
        cond = [
            [
                torch.zeros(1, 2, 3),
                {"minimax_keyframes": [old1], "minimax_refs": ref, "marker": "first"},
            ],
            [torch.zeros(1, 2, 3), {"minimax_keyframes": [old2], "marker": "second"}],
        ]
        out = Guide.execute(
            cond, latent(), RecordingVAEDouble(), 120, 0, image(), image(offset=0.1)
        ).result[0]
        self.assertIs(out[0][1]["minimax_refs"], ref)
        self.assertIs(out[0][1]["minimax_keyframes"][0], old1)
        self.assertIs(out[1][1]["minimax_keyframes"][0], old2)
        self.assertEqual(
            [r["resolved_frame_index"] for r in out[0][1]["minimax_keyframes"]],
            [40, 0, 119],
        )
        self.assertEqual(
            [r["resolved_frame_index"] for r in out[1][1]["minimax_keyframes"]],
            [60, 0, 119],
        )
        self.assertEqual(len(cond[0][1]["minimax_keyframes"]), 1)

    def test_native_layout_and_condition_payload_keep_mixed_references(self):
        from comfy.ldm.minimax.model import PackedLayout
        from comfy.model_base import MiniMaxH3

        references = [
            {
                "kind": "image",
                "latent_h": 2,
                "latent_w": 2,
                "latent": torch.ones(1, 24, 1, 2, 2),
            },
            {
                "kind": "audio",
                "ref_audio_t": 3,
                "audio_latent": torch.ones(1, 32, 2, 3),
            },
            {
                "kind": "video_audio",
                "latent_t": 2,
                "latent_h": 2,
                "latent_w": 2,
                "ref_audio_t": 4,
                "latent": torch.ones(1, 24, 2, 2, 2),
                "audio_latent": torch.ones(1, 32, 2, 4),
            },
        ]
        existing_visual = {
            "resolved_frame_index": 40,
            "latent": torch.ones(1, 24, 1, 4, 4),
        }
        existing_audio = {
            "resolved_frame_index": 0,
            "audio_latent": torch.ones(1, 32, 2, 2),
        }
        conditioning = positive(
            {
                "minimax_refs": references,
                "minimax_keyframes": [existing_visual, existing_audio],
            }
        )
        av_latent = latent(w=64, h=64)
        result = Guide.execute(
            conditioning,
            av_latent,
            RecordingVAEDouble(),
            120,
            0,
            image(),
            image(offset=0.1),
        ).result[0]
        metadata = result[0][1]
        guides = metadata["minimax_keyframes"]
        self.assertIs(metadata["minimax_refs"], references)

        # Exercise the model's real payload assembly without constructing a model.
        payload = MiniMaxH3._cond_latents(guides, metadata["minimax_refs"])
        expected_video = [
            existing_visual["latent"],
            guides[2]["latent"],
            guides[3]["latent"],
            references[0]["latent"],
            references[2]["latent"],
        ]
        expected_audio = [
            existing_audio["audio_latent"],
            references[1]["audio_latent"],
            references[2]["audio_latent"],
        ]
        self.assertEqual(len(payload["cond_video_latents"]), len(expected_video))
        self.assertEqual(len(payload["cond_audio_latents"]), len(expected_audio))
        for actual, expected in zip(payload["cond_video_latents"], expected_video):
            self.assertIs(actual, expected)
        for actual, expected in zip(payload["cond_audio_latents"], expected_audio):
            self.assertIs(actual, expected)

        video, audio = av_latent["samples"].tensors
        layout = PackedLayout(
            text_len=4,
            latent_t=video.shape[2],
            latent_h=video.shape[3],
            latent_w=video.shape[4],
            audio_t=audio.shape[-1],
            keyframes=guides,
            refs=metadata["minimax_refs"],
        )
        self.assertEqual(
            [kind for _, _, kind in layout.segments],
            [
                "text",
                "cond",
                "cond_audio",
                "cond",
                "cond",
                "ref_img",
                "ref_audio",
                "ref_audio",
                "ref_img",
                "audio",
                "video",
            ],
        )
        self.assertEqual(layout.position_ids.shape, (layout.seq_len, 3))
        self.assertEqual(layout.img_pos.numel(), 3 * 4 + 1 + 2 + video.shape[2] * 4)

    def test_overlapping_visual_and_wrong_canvas_rejected(self):
        for existing in [
            {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 3, 4)},
            {"resolved_frame_index": 118, "latent": torch.zeros(1, 24, 2, 3, 4)},
            {"resolved_frame_index": 40, "latent": torch.zeros(1, 24, 1, 2, 2)},
        ]:
            with self.subTest(existing=list(existing["latent"].shape)):
                with self.assertRaises(ValueError):
                    Guide.execute(
                        positive({"minimax_keyframes": [existing]}),
                        latent(),
                        RecordingVAEDouble(),
                        120,
                        0,
                        image(),
                        image(),
                    )
        audio = {"resolved_frame_index": 0, "audio_latent": torch.ones(1, 32, 2, 1)}
        out = Guide.execute(
            positive({"minimax_keyframes": [audio]}),
            latent(),
            RecordingVAEDouble(),
            120,
            0,
            image(),
        )
        self.assertEqual(len(keyframes(out)), 2)

    def test_one_frame_equal_rgb_deduplicated_and_conflict_rejected(self):
        f = image()
        rgba = torch.cat([f, torch.zeros_like(f[..., :1])], -1)
        vae = RecordingVAEDouble()
        out = Guide.execute(positive(), latent(), vae, 1, 5, f, rgba)
        self.assertEqual(len(keyframes(out)), 1)
        self.assertEqual(len(vae.calls), 1)
        with self.assertRaises(ValueError):
            Guide.execute(positive(), latent(), vae, 1, 5, f, image(offset=0.2))
        self.assertTrue(torch.equal(Pin.execute(f, f, rgba).result[0], f))
        with self.assertRaises(ValueError):
            Pin.execute(f, f, image(offset=0.2))

    def test_invalid_endpoint_inputs_and_windows(self):
        bad = [
            torch.zeros(2, 32, 48, 3),
            torch.zeros(1, 32, 48, 1),
            torch.zeros(32, 48, 3),
            torch.zeros(1, 0, 48, 3),
            torch.zeros(1, 32, 48, 3, dtype=torch.uint8),
        ]
        for value in bad:
            with self.subTest(shape=list(value.shape)):
                with self.assertRaises(ValueError):
                    Guide.execute(
                        positive(), latent(), RecordingVAEDouble(), 120, 0, value
                    )
                with self.assertRaises(ValueError):
                    Pin.execute(torch.zeros(5, 32, 48, 3), value)
        for length, offset in [
            (0, 0),
            (-1, 0),
            (1, -1),
            (1.5, 0),
            (True, 0),
            (120, 1.5),
        ]:
            with self.subTest(length=length, offset=offset):
                with self.assertRaises(ValueError):
                    Guide.execute(
                        positive(),
                        latent(),
                        RecordingVAEDouble(),
                        length,
                        offset,
                        image(),
                    )
        for wrong in [
            {"samples": torch.zeros(1, 24, 37, 3, 4)},
            {
                "samples": comfy.nested_tensor.NestedTensor(
                    (torch.zeros(1, 23, 37, 3, 4), torch.zeros(1, 32, 2, 207))
                )
            },
        ]:
            with self.assertRaises(ValueError):
                Guide.execute(positive(), wrong, RecordingVAEDouble(), 120, 0, image())

    def test_pin_default_single_dual_immutable_and_middle_exact(self):
        batch = torch.linspace(0, 1, 7 * 32 * 48 * 3).reshape(7, 32, 48, 3)
        original = batch.clone()
        self.assertIs(Pin.execute(batch).result[0], batch)
        f, l = (image(offset=0.05), image(offset=0.15))
        for first, last in [(f, None), (None, l), (f, l)]:
            with self.subTest(first=first is not None, last=last is not None):
                out = Pin.execute(batch, first, last).result[0]
                self.assertIsNot(out, batch)
                self.assertTrue(torch.equal(batch, original))
                self.assertTrue(torch.equal(out[1:-1], batch[1:-1]))
                self.assertTrue(
                    torch.equal(out[0], f[0] if first is not None else batch[0])
                )
                self.assertTrue(
                    torch.equal(out[-1], l[0] if last is not None else batch[-1])
                )

    def test_pin_center_lanczos_and_rgba(self):
        batch = torch.rand(6, 48, 64, 3)
        anchor = image(29, 73, channels=4)
        out = Pin.execute(batch, anchor, anchor).result[0]
        expected = comfy.utils.common_upscale(
            anchor[..., :3].movedim(-1, 1), 64, 48, "lanczos", "center"
        ).movedim(1, -1)
        self.assertTrue(torch.equal(out[0], expected[0]))
        self.assertTrue(torch.equal(out[-1], expected[0]))
        self.assertTrue(torch.equal(out[1:-1], batch[1:-1]))

    def test_ffmpeg_h264_roundtrip_is_lossy_but_pins_survive(self):
        missing_tools = [
            tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)
        ]
        if missing_tools:
            self.skipTest("Export test requires: " + ", ".join(missing_tools))
        h, w, n = (48, 64, 9)
        batch = torch.linspace(0, 0.8, n * h * w * 3).reshape(n, h, w, 3)
        first = torch.zeros(1, h, w, 3)
        first[..., 0] = 0.9
        first[..., 2] = 0.8
        last = torch.zeros(1, h, w, 3)
        last[..., 1] = 0.9
        out = Pin.execute(batch, first, last).result[0]
        self.assertTrue(torch.equal(out[1:-1], batch[1:-1]))
        rgb = (out.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        with tempfile.TemporaryDirectory(prefix="h3-endpoint-export-") as tmp:
            movie = Path(tmp) / "pins.mp4"
            enc = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    "24",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(movie),
                ],
                input=rgb.tobytes(),
                capture_output=True,
            )
            self.assertEqual(enc.returncode, 0, enc.stderr.decode())
            dec = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(movie),
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ],
                capture_output=True,
            )
            self.assertEqual(dec.returncode, 0, dec.stderr.decode())
            decoded = np.frombuffer(dec.stdout, np.uint8).reshape(n, h, w, 3)
            errors = np.abs(decoded.astype(np.int16) - rgb.astype(np.int16)).mean(
                axis=(1, 2, 3)
            )
            self.assertLess(float(errors[0]), 5)
            self.assertLess(float(errors[-1]), 5)
            self.assertGreater(float(errors.max()), 0)
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "stream=codec_name,nb_frames,r_frame_rate,pix_fmt",
                        "-of",
                        "json",
                        str(movie),
                    ]
                )
            )
            self.assertEqual(probe["streams"][0]["codec_name"], "h264")
            self.assertEqual(probe["streams"][0]["pix_fmt"], "yuv420p")
            self.assertEqual(probe["streams"][0]["r_frame_rate"], "24/1")
            self.assertEqual(int(probe["streams"][0]["nb_frames"]), n)

    def test_real_validate_prompt_and_cpu_graph_execution(self):
        captured = {}

        class Source(io.ComfyNode):

            @classmethod
            def define_schema(cls):
                return io.Schema(
                    node_id="EndpointTestSource",
                    inputs=[],
                    outputs=[
                        io.Conditioning.Output(),
                        io.Latent.Output(),
                        io.Vae.Output(),
                        io.Image.Output(),
                        io.Image.Output(),
                        io.Image.Output(),
                    ],
                )

            @classmethod
            def execute(cls):
                return io.NodeOutput(
                    positive(),
                    latent(141),
                    RecordingVAEDouble(),
                    torch.zeros(120, 48, 64, 3),
                    image(48, 64),
                    image(48, 64, offset=0.1),
                )

        class Sink(io.ComfyNode):

            @classmethod
            def define_schema(cls):
                return io.Schema(
                    node_id="EndpointTestSink",
                    is_output_node=True,
                    inputs=[
                        io.Conditioning.Input("positive"),
                        io.Image.Input("images"),
                    ],
                    outputs=[],
                )

            @classmethod
            def execute(cls, positive, images):
                captured["anchors"] = [
                    x["resolved_frame_index"]
                    for x in positive[0][1]["minimax_keyframes"]
                ]
                captured["shape"] = list(images.shape)
                captured["middle_zero"] = bool(torch.count_nonzero(images[1:-1]) == 0)
                return io.NodeOutput()

        additions = {
            **module.NODE_CLASS_MAPPINGS,
            "EndpointTestSource": Source,
            "EndpointTestSink": Sink,
        }
        previous = {k: nodes.NODE_CLASS_MAPPINGS.get(k) for k in additions}
        nodes.NODE_CLASS_MAPPINGS.update(additions)
        prompt = {
            "source": {"class_type": "EndpointTestSource", "inputs": {}},
            "guide": {
                "class_type": "H3ClipEndpointGuide",
                "inputs": {
                    "positive": ["source", 0],
                    "latent": ["source", 1],
                    "vae": ["source", 2],
                    "visible_frames": 120,
                    "trim_start": 5,
                    "first_frame": ["source", 4],
                    "last_frame": ["source", 5],
                },
            },
            "pin": {
                "class_type": "H3PinClipEndpoints",
                "inputs": {
                    "images": ["source", 3],
                    "first_frame": ["source", 4],
                    "last_frame": ["source", 5],
                },
            },
            "sink": {
                "class_type": "EndpointTestSink",
                "inputs": {"positive": ["guide", 0], "images": ["pin", 0]},
            },
        }
        try:
            valid, error, outputs, node_errors = asyncio.run(
                execution.validate_prompt("h3-endpoint-test", prompt, None)
            )
            self.assertTrue(valid, (error, node_errors))
            server = SimpleNamespace(client_id=None, send_sync=Mock())
            executor = execution.PromptExecutor(
                server,
                cache_type=execution.CacheType.NONE,
                cache_args={"ram": 0, "ram_inactive": 0},
            )
            executor.execute(prompt, "h3-endpoint-test", execute_outputs=outputs)
            self.assertTrue(executor.success, executor.status_messages)
            self.assertEqual(
                captured,
                {"anchors": [5, 124], "shape": [120, 48, 64, 3], "middle_zero": True},
            )
            invalid = copy.deepcopy(prompt)
            invalid["pin"]["inputs"]["first_frame"] = ["source", 1]
            with self.assertLogs(level="ERROR"):
                bad = asyncio.run(execution.validate_prompt("bad-type", invalid, None))
            self.assertFalse(bad[0])
        finally:
            for k, old in previous.items():
                if old is None:
                    nodes.NODE_CLASS_MAPPINGS.pop(k, None)
                else:
                    nodes.NODE_CLASS_MAPPINGS[k] = old


if __name__ == "__main__":
    unittest.main(argv=[argument for argument in sys.argv if argument != "--cpu"])
