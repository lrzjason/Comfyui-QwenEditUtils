import node_helpers
import comfy.utils
import math
from PIL import Image
import numpy as np
import torch

RESOLUTION_CONFIG = {
    # 2048: [
    #     (2048, 2048),        # 1024*2, 1024*2
    #     (1344, 3136),        # 672*2, 1568*2
    #     (1408, 3008),        # 704*2, 1504*2
    #     (1472, 2944),        # 736*2, 1472*2
    #     (1536, 2816),        # 768*2, 1408*2
    #     (1600, 2688),        # 800*2, 1344*2
    #     (1664, 2496),        # 832*2, 1248*2
    #     (1792, 2368),        # 896*2, 1184*2
    #     (1920, 2240),        # 960*2, 1120*2
    # ],
    # 1536: [
    #     (1536, 1536),        # 1024*1.5, 1024*1.5
    #     (1008, 2352),        # 672*1.5, 1568*1.5
    #     (1056, 2256),        # 704*1.5, 1504*1.5
    #     (1104, 2208),        # 736*1.5, 1472*1.5
    #     (1152, 2112),        # 768*1.5, 1408*1.5
    #     (1200, 2016),        # 800*1.5, 1344*1.5
    #     (1248, 1872),        # 832*1.5, 1248*1.5
    #     (1344, 1776),        # 896*1.5, 1184*1.5
    #     (1440, 1680),        # 960*1.5, 1120*1.5
    # ],
    # 1328: [
    #     (1328, 1328),
    #     (880, 2032),
    #     (912, 1952),
    #     (960, 1904),
    #     (992, 1824),
    #     (1040, 1744),
    #     (1072, 1616),
    #     (1168, 1536),
    #     (1248, 1456),
    # ],
    1024: [
        (512, 1280),
        (576, 1408),
        (640, 1536),
        (672, 1568),
        (640, 1408),
        (704, 1504),
        (736, 1472),
        (768, 1408),
        (800, 1344),
        (832, 1344),
        (896, 1408),
        (832, 1248),
        (960, 1280),
        (896, 1184),
        (1024, 1280),
        (960, 1120),
        (1024, 1152),
        (1088, 1152),
        (1024, 1024)
    ]
    # 768: [
    #     (768, 768),
    #     (512, 1184),
    #     (512, 1152),
    #     (544, 1088),
    #     (576, 1056),
    #     (608, 992),
    #     (640, 960),
    #     (672, 896),
    #     (704, 832),
    # ],
    # # based on 1024 to create 512
    # 512: [
    #     (512, 512),
    #     (352, 800),
    #     (352, 768),
    #     (384, 736),
    #     (384, 704),
    #     (416, 672),
    #     (416, 640),
    #     (448, 608),
    #     (480, 576),
    # ],
}

def get_nearest_resolution(image, resolution=1024):
    height, width, _ = image.shape
    resolution_set = RESOLUTION_CONFIG[resolution]
    
    # get ratio
    image_ratio = width / height

    target_set = resolution_set.copy()
    reversed_set = [(y, x) for x, y in target_set]
    target_set = sorted(set(target_set + reversed_set))
    target_ratio = list(set([round(width/height, 2) for width,height in target_set]))
    
    # Find the closest vertical ratio
    closest_ratio = min(target_ratio, key=lambda x: abs(x - image_ratio))
    closest_resolution = target_set[target_ratio.index(closest_ratio)]

    return closest_ratio,closest_resolution


def crop_image(image,resolution):
    height, width, _ = image.shape
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
    scale_ratio = closest_resolution[0] / closest_resolution[1]
    image_ratio = width / height
    scale_with_height = True
    if image_ratio < scale_ratio: 
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
    print("ori size:",height,width)
    if scale_with_height: 
        up_scale = height / closest_resolution[1]
    else:
        up_scale = width / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = abs(expanded_closest_size[0] - width)
    diff_y = abs(expanded_closest_size[1] - height)

    crop_x = 0
    crop_y = 0
    # crop extra part of the resized images
    if diff_x>0:
        crop_x =  diff_x //2
        cropped_image = image[:,  crop_x:width-diff_x+crop_x, :]
    elif diff_y>0:
        crop_y =  diff_y//2
        cropped_image = image[crop_y:height-diff_y+crop_y, :, :]
    else:
        # 1:1 ratio
        cropped_image = image

    height, width, _ = cropped_image.shape  
    f_width, f_height = closest_resolution
    cropped_image = convert_float_unit8(cropped_image)
    print("cropped_image:",cropped_image)
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
                print("cropped_image:",cropped_image.shape)
                print("cropped_image:",cropped_image)
                
            images = [image]
            if vae is not None:
                ref_latent = vae.encode(image)
                print("ref_latent:",ref_latent.shape)
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