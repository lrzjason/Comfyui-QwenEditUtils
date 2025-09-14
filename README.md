# Comfyui-QwenEditUtils

A collection of utility nodes for Qwen-based image editing in ComfyUI.

## Node

![Example](https://github.com/lrzjason/Comfyui-QwenEditUtils/blob/master/example.png?timestep=1678165125)

### TextEncodeQwenImageEdit 小志Jason(xiaozhijason)

This node provides text encoding functionality with reference image support for Qwen-based image editing workflows. It allows you to encode prompts while incorporating reference images for more controlled image generation.

#### Inputs

- **clip**: The CLIP model to use for encoding
- **prompt**: The text prompt to encode
- **vae** (optional): The VAE model for image encoding
- **image** (optional): Reference image for image editing
- **enable_resize** (optional): Enable automatic resizing of the reference image
- **resolution** (optional): Target resolution for image resizing (512, 768, 1024, 1328, 1536, 2048)

#### Outputs

- **CONDITIONING**: The encoded conditioning tensor
- **IMAGE**: The processed reference image
- **LATENT**: The encoded latent representation of the reference image

#### Behavior

- Encodes text prompts using CLIP with optional reference image guidance
- Automatically resizes reference images to optimal dimensions based on the selected resolution
- Supports various resolution presets for different generation needs
- Integrates with VAE models to encode reference images into latent space

## Key Features

- **Reference Image Support**: Incorporate reference images into your text-to-image generation workflow
- **Automatic Image Resizing**: Automatically resize reference images to optimal dimensions
- **Multiple Resolution Presets**: Choose from various resolution options (512 to 2048)
- **Latent Space Integration**: Encode reference images into latent space for efficient processing
- **Qwen Model Compatibility**: Specifically designed for Qwen-based image editing models

## Resolution Recommendations

The recommended resolution is **1024**, which provides the best balance of quality and performance for most use cases.

**Performance Warning**: Using resolutions other than 1024 may result in:
- Reduced generation quality
- Slower processing times
- Higher memory consumption
- Unstable behavior in some cases

Choose alternative resolutions only when you have specific requirements that cannot be met with the 1024 preset.

## Installation

1. Clone or download this repository into your ComfyUI's `custom_nodes` directory.
2. Restart ComfyUI.
3. The node will be available in the "advanced/conditioning" category.

## Usage

1. Add the "TextEncodeQwenImageEdit 小志Jason(xiaozhijason)" node to your workflow.
2. Connect a CLIP model to the clip input.
3. Enter your text prompt in the prompt field.
4. Optionally, connect a reference image to the image input.
5. Configure the resolution and enable_resize options as needed.
6. Connect the outputs to your image generation nodes.

## Contact
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)  
- **Email**: lrzjason@gmail.com  
- **QQ Group**: 866612947  
- **Wechatid**: fkdeai
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)


## Sponsors me for more open source projects:
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>Buy me a coffee:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/bmc_qr.png" alt="Buy Me a Coffee QR" width="200" />
      </td>
      <td align="center">
        <p>WeChat:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/wechat.jpg" alt="WeChat QR" width="200" />
      </td>
    </tr>
  </table>
</div>