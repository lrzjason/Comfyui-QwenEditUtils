import node_helpers
import comfy.utils
import math
from PIL import Image
import numpy as np
import torch
import cv2
import copy

def get_nearest_resolution(image, resolution=1024):
    height, width, _ = image.shape
    
    # get ratio
    image_ratio = width / height

    # Calculate target dimensions that:
    # 1. Maintain the aspect ratio
    # 2. Have an area of approximately resolution^2 (1024*1024 = 1048576)
    # 3. Are divisible by 8
    target_area = resolution * resolution
    
    # width = height * image_ratio
    # width * height = target_area
    # height * image_ratio * height = target_area
    # height^2 = target_area / image_ratio
    height_optimal = math.sqrt(target_area / image_ratio)
    width_optimal = height_optimal * image_ratio
    
    # Round to nearest multiples of 8
    height_8 = int(height_optimal / 8 + 0.5) * 8
    width_8 = int(width_optimal / 8 + 0.5) * 8
    
    # Ensure minimum size of 64x64
    height_8 = max(64, height_8)
    width_8 = max(64, width_8)
    
    closest_resolution = (width_8, height_8)
    closest_ratio = width_8 / height_8

    return closest_ratio, closest_resolution


def crop_image(image, resolution, resize_method="cv2"):
    height, width, _ = image.shape
    closest_ratio, closest_resolution = get_nearest_resolution(image, resolution=resolution)
    image_ratio = width / height
    
    # Determine which dimension to scale by to minimize cropping (more efficiently)
    scale_with_height = image_ratio >= closest_ratio
    
    return simple_center_crop(image, scale_with_height, closest_resolution, resize_method)[0]
def simple_center_crop(image, scale_with_height, closest_resolution, resize_method="cv2"):
    height, width, _ = image.shape
    # print("ori size:",height,width)
    if scale_with_height: 
        # Scale based on height, then crop width if needed
        up_scale = height / closest_resolution[1]
    else:
        # Scale based on width, then crop height if needed
        up_scale = width / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = expanded_closest_size[0] - width
    diff_y = expanded_closest_size[1] - height

    crop_x = 0
    crop_y = 0
    # crop extra part of the resized images
    if diff_x > 0:
        # Need to crop width (image is wider than needed)
        crop_x = diff_x // 2
        cropped_image = image[:, crop_x:width - diff_x + crop_x, :]
    elif diff_y > 0:
        # Need to crop height (image is taller than needed)
        crop_y = diff_y // 2
        cropped_image = image[crop_y:height - diff_y + crop_y, :, :]
    else:
        # No cropping needed
        cropped_image = image

    f_width, f_height = closest_resolution
    
    # Convert to uint8 for processing
    cropped_image_uint8 = (cropped_image * 255).astype(np.uint8)
    
    # Use faster resize method with cv2 if available, otherwise use PIL
    if resize_method == "cv2":
        resized_img = cv2.resize(cropped_image_uint8, (f_width, f_height), interpolation=cv2.INTER_AREA)
    else:
        # Fall back to PIL if cv2 is not available
        img_pil = Image.fromarray(cropped_image_uint8)
        resized_img = img_pil.resize((f_width, f_height), Image.LANCZOS)
        resized_img = np.array(resized_img)
    
    # Convert back to float32
    resized_img = resized_img.astype(np.float32) / 255.0
    return resized_img, crop_x, crop_y


class TextEncodeQwenImageEdit_lrzjason:
    @classmethod
    def INPUT_TYPES(s):
        resolution_choices = [
            2048, 1536, 1328, 1024, 768, 512
        ]
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image": ("IMAGE", ),
                "enable_resize": ("BOOLEAN", {"default": True}),
                "resolution": (resolution_choices, {
                    "default": 1024,
                }),
                "resize_method": (["cv2", "pil"], {
                    "default": "cv2",
                }),
                "vl_encode_resize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "LATENT", )
    RETURN_NAMES = ("conditioning", "cropped_image", "latent")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, image=None, enable_resize=True, resolution=1024, resize_method="cv2", vl_encode_resize=False):
        ref_latent = None
        images = []
        vae_image = None
        if image is not None:
            # Process image if needed
            samples = image.squeeze(0).numpy()  # Convert to HWC format
            if enable_resize:
                # More efficient processing
                cropped_image = crop_image(samples, resolution, resize_method)
                image = torch.from_numpy(cropped_image).unsqueeze(0)  # Convert back to BCHW format
                
            if vae is not None:
                ref_latent = vae.encode(image)
                vae_image = image.clone()
                
            if vl_encode_resize:
                cropped_image = crop_image(samples, 384, resize_method)
                image = torch.from_numpy(cropped_image).unsqueeze(0)  # Convert back to BCHW format
            images = [image]
                
        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]})
            
        return (conditioning, vae_image, {"samples": ref_latent})


class TextEncodeQwenImageEditAdvanced_lrzjason:
    @classmethod
    def INPUT_TYPES(s):
        resolution_choices = [
            2048, 1536, 1328, 1024, 768, 512
        ]
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image": ("IMAGE", ),
                "enable_resize": ("BOOLEAN", {"default": True}),
                "resolution": (resolution_choices, {
                    "default": 1024,
                }),
                "resize_method": (["cv2", "pil"], {
                    "default": "cv2",
                }),
                "return_cond_without_image": ("BOOLEAN", {"default": True}),
                "vl_encode_resize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "LATENT", )
    RETURN_NAMES = ("conditioning", "cond_without_image", "cropped_image", "latent")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, image=None, enable_resize=True, resolution=1024, resize_method="cv2", return_cond_without_image=True, vl_encode_resize=False):
        ref_latent = None
        images = []
        vae_image = None
        if image is not None:
            # Process image if needed
            samples = image.squeeze(0).numpy()  # Convert to HWC format
            if enable_resize:
                # More efficient processing
                cropped_image = crop_image(samples, resolution, resize_method)
                image = torch.from_numpy(cropped_image).unsqueeze(0)  # Convert back to BCHW format
                
            if vae is not None:
                ref_latent = vae.encode(image)
                vae_image = image.clone()
                
            if vl_encode_resize:
                cropped_image = crop_image(samples, 384, resize_method)
                image = torch.from_numpy(cropped_image).unsqueeze(0)  # Convert back to BCHW format
            images = [image]
                
        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]})
        
        conditioning_without_images = None
        if return_cond_without_image:
            tokens_without_images = clip.tokenize(prompt)
            conditioning_without_images = clip.encode_from_tokens_scheduled(tokens_without_images)
            
        return (conditioning, conditioning_without_images, vae_image, {"samples": ref_latent})

NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEdit_lrzjason": TextEncodeQwenImageEdit_lrzjason,
    "TextEncodeQwenImageEditAdvanced_lrzjason": TextEncodeQwenImageEditAdvanced_lrzjason
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEdit_lrzjason": "TextEncodeQwenImageEdit 小志Jason(xiaozhijason)",
    "TextEncodeQwenImageEditAdvanced_lrzjason": "TextEncodeQwenImageEditAdvanced 小志Jason(xiaozhijason)",
}