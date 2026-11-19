# ComfyUI人物替换工作流

用一张角色参考图替换源视频中的人物。本仓库提供 [海螺 H3 人物替换专用流 v2.51](./海螺H3+人物替换专用流v2.51.json)、配套自定义节点、7 个模型文件，以及用于编写人物替换提示词的技能包。以下步骤适用于 Windows 上的 Comfy Desktop。

## 安装与导入

1. **安装 ComfyUI。** 下载并安装 Comfy Desktop，首次启动时选择在本机运行。此时可以先跳过模型配置。[下载 Comfy Desktop](https://comfy.org/download)
2. **新建项目并导入工作流。** 在 Comfy Desktop 中创建一个本地 ComfyUI 项目库，启动项目并打开画布。用下面的命令下载本仓库和大文件，然后将仓库根目录中的 `海螺H3+人物替换专用流v2.51.json` 直接拖入画布。

   ```powershell
   git lfs install
   git clone https://github.com/LoveWood233/ComfyUI-H3-Subject-Replace.git
   cd ComfyUI-H3-Subject-Replace
   git lfs pull
   ```

   上述命令需要 [Git](https://git-scm.com/downloads/win) 和 [Git LFS](https://git-lfs.com/)。如果模型文件只有百余字节，说明本地拿到的是 LFS 指针，请在克隆的仓库中运行 `git lfs pull`。

3. **找到 Comfy Desktop 数据目录。** 打开创建项目时选定的 `Comfy-Desktop` 文件夹。正确位置下面同时有 `ComfyUI-Shared` 和 `ComfyUI-Installs`。使用默认位置时，可在资源管理器地址栏输入 `%LOCALAPPDATA%\Comfy-Desktop`。[Windows 安装目录说明](https://docs.comfy.org/installation/desktop/windows)
4. **导入模型。** 将仓库中的整个 `ComfyUI-Shared` 文件夹复制到上述 `Comfy-Desktop` 目录，与同名文件夹合并。模型应落在 `Comfy-Desktop\ComfyUI-Shared\models\` 下。首次安装还需补齐下表列出的 4 个基础权重，已有对应文件的项目可直接复用。
5. **导入自定义节点。** 打开 `Comfy-Desktop\ComfyUI-Installs\<项目名>\ComfyUI\custom_nodes\`。把仓库 `ComfyUI-Installs\custom_nodes\` **里面的插件文件夹**复制到这里。以 VideoHelperSuite 为例，复制后的路径应为 `<项目名>\ComfyUI\custom_nodes\comfyui-videohelpersuite\`。如启动日志提示缺少 Python 包，可在该项目环境中按对应插件的 `requirements.txt` 安装依赖。
6. **重启项目。** 回到 ComfyUI 界面，点击上方的**项目名称 → 重启项目**。重新打开工作流，确认模型下拉框能选中对应文件。将人物参考图放入节点 **215**，源视频放入 **286**，提示词填入 **248**，即可运行短片替换。

## 首次运行需要补齐的模型

仓库内的 7 个权重合计约 **4.87 GiB**，包括 LoRA、潜空间放大模型和预览模型。原工作流的短片分支还引用下面 4 个文件，合计约 **37.32 GiB**。下载后按下表放入 `ComfyUI-Shared\models\` 下的对应子目录。

| 节点 | 子目录 | 文件与下载来源 |
| --- | --- | --- |
| 211 · 混合主模型 | `diffusion_models\minimax\` | [minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/blob/main/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors) |
| 210 · 文本编码器 | `text_encoders\minimax\` | [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| 204 · 视频 VAE | `vae\` | [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) |
| 188 · 音频 VAE | `vae\` | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) |

**两个 LoRA 加载位。** 仓库将 LoRA 放在 `models\loras\`，工作流默认查找 `loras\minimaxh3\`。重启后，在节点 **213** 选择 `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`，在节点 **285** 选择 `h3_character_swap_v1.5.safetensors`。如果要沿用原 JSON 的默认选择，可在第 4 步把这两个文件放进 `models\loras\minimaxh3\`。

**源视频和长片分支。** 原工作流默认画幅是 9:16，横版素材请在节点 **209** 选择对应比例。源视频没有音轨时，请断开视频加载器到参考条件和视频输出节点的音频连线。工作流作者建议超过 7 秒的素材分段处理；如启用右侧的长片循环分支，还需配置该分支引用的 Easy-Use 循环节点和额外的放大模型。
