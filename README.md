# Comfyui-QwenEditUtils

A collection of utility nodes for Qwen-based image editing in ComfyUI.

## Node

### TextEncodeQwenImageEditPlus 小志Jason(xiaozhijason)

This node provides text encoding functionality with reference image support for Qwen-based image editing workflows. It allows you to encode prompts while incorporating up to 5 reference images for more controlled image generation.

#### Inputs

- **clip**: The CLIP model to use for encoding
- **prompt**: The text prompt to encode
- **vae** (optional): The VAE model for image encoding
- **image1** (optional): First reference image for image editing
- **image2** (optional): Second reference image for image editing
- **image3** (optional): Third reference image for image editing
- **image4** (optional): Fourth reference image for image editing
- **image5** (optional): Fifth reference image for image editing
- **enable_resize** (optional): Enable automatic resizing of the reference image
- **llama_template** (optional): Custom Llama template for image description and editing instructions

#### Outputs

- **CONDITIONING**: The encoded conditioning tensor
- **IMAGE**: The processed reference images
- **LATENT**: The encoded latent representation of the first reference image

#### Behavior

- Encodes text prompts using CLIP with optional reference image guidance
- Supports up to 5 reference images for complex editing tasks
- Automatically resizes reference images to optimal dimensions
- Integrates with VAE models to encode reference images into latent space
- Supports custom Llama templates for more precise image editing instructions

## Key Features

- **Multi-Image Support**: Incorporate up to 5 reference images into your text-to-image generation workflow
- **Automatic Image Resizing**: Automatically resize reference images to optimal dimensions
- **Latent Space Integration**: Encode reference images into latent space for efficient processing
- **Qwen Model Compatibility**: Specifically designed for Qwen-based image editing models
- **Customizable Templates**: Use custom Llama templates for tailored image editing instructions

## Installation

1. Clone or download this repository into your ComfyUI's `custom_nodes` directory.
2. Restart ComfyUI.
3. The node will be available in the "advanced/conditioning" category.

## Usage

1. Add the "TextEncodeQwenImageEditPlus 小志Jason(xiaozhijason)" node to your workflow.
2. Connect a CLIP model to the clip input.
3. Enter your text prompt in the prompt field.
4. Optionally, connect up to 5 reference images to the image inputs.
5. Configure the enable_resize and other options as needed.
6. Connect the outputs to your image generation nodes.

## Update Log

### v1.0.3
- Fixed critical bug with undefined `image_prompt` variable
- Fixed error when no reference images are provided
- Improved node stability and error handling
- Updated node implementation to support up to 5 reference images
- Added support for custom Llama templates
- Improved image processing and resizing logic
- Enhanced VL encoding with better image description capabilities

### v1.0.2
- Updated node implementation to support up to 5 reference images
- Added support for custom Llama templates
- Improved image processing and resizing logic
- Enhanced VL encoding with better image description capabilities
- Fixed issues with latent encoding for multiple images

### v1.0.1
- Initial release with basic text encoding and single image reference support

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
