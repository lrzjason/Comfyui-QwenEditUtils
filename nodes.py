import node_helpers
import comfy.utils
import math
from PIL import Image
import numpy as np
import torch

# 
# RESOLUTION_CONFIG = {
#     2048: [
#         (1024, 2560),   # 1 ← 512x1280
#         (1152, 2816),   # 2 ← 576x1408
#         (1280, 3072),   # 3 ← 640x1536
#         (1280, 2816),   # 4 ← 640x1408
#         (1408, 3008),   # 5 ← 704x1504 → 3008 ÷ 64 = 47 ✅
#         (1536, 2816),   # 6 ← 768x1408
#         (1664, 2688),   # 7 ← 832x1344
#         (1792, 2368),   # 8 ← 896x1184
#         (1920, 2240),   # 9 ← 960x1120
#         (2048, 2560),   # 10 ← 1024x1280
#         (2176, 2304),   # 11 ← 1088x1152
#         (2048, 2048),   # 12 ← 1024x1024
#         (2048, 2304),   # 13 ← 1024x1152
#     ],
#     1536: [
#         (768, 1920),    # 1 ← 512x1280
#         (896, 2112),    # 2 ← 576x1408
#         (960, 2304),    # 3 ← 640x1536
#         (960, 2112),    # 4 ← 640x1408
#         (1056, 2240),   # 5 ← 704x1504
#         (1152, 2112),   # 6 ← 768x1408
#         (1248, 2048),   # 7 ← 832x1344
#         (1344, 1792),   # 8 ← 896x1184
#         (1408, 1664),   # 9 ← 960x1120 ← FIXED: was (1440,1664)
#         (1536, 1920),   # 10 ← 1024x1280
#         (1632, 1728),   # 11 ← 1088x1152
#         (1536, 1536),   # 12 ← 1024x1024
#         (1536, 1728),   # 13 ← 1024x1152
#     ],
#     1328: [
#         (640, 1664),    # 1 ← 512x1280
#         (768, 1856),    # 2 ← 576x1408
#         (832, 1984),    # 3 ← 640x1536
#         (832, 1856),    # 4 ← 640x1408
#         (896, 1920),    # 5 ← 704x1504
#         (1024, 1856),   # 6 ← 768x1408
#         (1088, 1728),   # 7 ← 832x1344
#         (1152, 1536),   # 8 ← 896x1184
#         (1280, 1472),   # 9 ← 960x1120 ← FIXED: was (1216,1472)
#         (1344, 1664),   # 10 ← 1024x1280
#         (1408, 1472),   # 11 ← 1088x1152
#         (1344, 1344),   # 12 ← 1024x1024
#         (1344, 1472),   # 13 ← 1024x1152
#     ],
#     1024: [
#         (512, 1280),    # 1
#         (576, 1408),    # 2
#         (640, 1536),    # 3
#         (640, 1408),    # 4
#         (704, 1504),    # 5
#         (768, 1408),    # 6
#         (832, 1344),    # 7
#         (896, 1184),    # 8
#         (960, 1120),    # 9
#         (1024, 1280),   # 10
#         (1088, 1152),   # 11
#         (1024, 1024),   # 12
#         (1024, 1152),   # 13
#     ],
#     768: [
#         (384, 960),     # 1 ← 512x1280
#         (448, 1056),    # 2 ← 576x1408
#         (512, 1152),    # 3 ← 640x1536
#         (512, 1056),    # 4 ← 640x1408
#         (512, 1088),    # 5 ← 704x1504 ← adjusted to avoid dup
#         (576, 1056),    # 6 ← 768x1408
#         (640, 1024),    # 7 ← 832x1344
#         (672, 896),     # 8 ← 896x1184
#         (704, 832),     # 9 ← 960x1120
#         (768, 960),     # 10 ← 1024x1280
#         (832, 896),     # 11 ← 1088x1152
#         (768, 768),     # 12 ← 1024x1024
#         (768, 832),     # 13 ← 1024x1152 ← adjusted to avoid dup
#     ],
#     512: [
#         (256, 640),     # 1 ← 512x1280
#         (320, 704),     # 2 ← 576x1408
#         (320, 768),     # 3 ← 640x1536
#         (320, 640),     # 4 ← 640x1408 ← adjusted to avoid dup
#         (384, 768),     # 5 ← 704x1504
#         (384, 704),     # 6 ← 768x1408
#         (448, 704),     # 7 ← 832x1344 ← FIXED: was (448,672)
#         (448, 576),     # 8 ← 896x1184
#         (512, 576),     # 9 ← 960x1120
#         (512, 640),     # 10 ← 1024x1280
#         (576, 576),     # 11 ← 1088x1152
#         (512, 512),     # 12 ← 1024x1024
#         (576, 640),     # 13 ← 1024x1152 ← adjusted to avoid dup
#     ],
# }

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
    height_8 = round(height_optimal / 8) * 8
    width_8 = round(width_optimal / 8) * 8
    
    # Ensure minimum size of 64x64
    height_8 = max(64, height_8)
    width_8 = max(64, width_8)
    
    closest_resolution = (width_8, height_8)
    closest_ratio = width_8 / height_8

    return closest_ratio, closest_resolution


def crop_image(image,resolution):
    height, width, _ = image.shape
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
    image_ratio = width / height
    
    # Determine which dimension to scale by to minimize cropping
    scale_with_height = True
    if image_ratio < closest_ratio: 
        scale_with_height = False
    
    try:
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
    except Exception as e:
        print(e)
        raise e
    return image

def convert_float_unit8(image):
    image = image.astype(np.float32) * 255
    return image.astype(np.uint8)

def convert_unit8_float(image):
    image = image.astype(np.float32)
    image = image / 255.
    return image
def simple_center_crop(image,scale_with_height,closest_resolution):
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

    height, width, _ = cropped_image.shape  
    f_width, f_height = closest_resolution
    cropped_image = convert_float_unit8(cropped_image)
    # print("cropped_image:",cropped_image)
    img_pil = Image.fromarray(cropped_image)
    resized_img = img_pil.resize((f_width, f_height), Image.LANCZOS)
    resized_img = np.array(resized_img)
    resized_img = convert_unit8_float(resized_img)
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
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "LATENT", )
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, image=None, enable_resize=True, resolution=1024):
        ref_latent = None
        if image is None:
            images = []
        else:
            # bs, h, w, c
            # ([1, 1248, 832, 3])
            if enable_resize:
                samples = image.squeeze(0).numpy()
                cropped_image = crop_image(samples,resolution)
                cropped_image = torch.from_numpy(cropped_image).unsqueeze(0)
                image = cropped_image
                # print("cropped_image:",cropped_image.shape)
                # print("cropped_image:",cropped_image)
                
            images = [image]
            if vae is not None:
                ref_latent = vae.encode(image)
                # print("ref_latent:",ref_latent.shape)
        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]})
            
        return (conditioning, image, {"samples":ref_latent}, )


NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEdit_lrzjason": TextEncodeQwenImageEdit_lrzjason,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEdit_lrzjason": "TextEncodeQwenImageEdit 小志Jason(xiaozhijason)",
}