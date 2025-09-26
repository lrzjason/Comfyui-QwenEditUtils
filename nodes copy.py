import node_helpers
import comfy.utils
import math
from PIL import Image
import numpy as np
import torch
import cv2
import copy


RESOLUTION_CONFIG = {
    384: [
        (192, 480),    # 1  ~0.40  (tall portrait)
        (216, 528),    # 2  ~0.41
        (240, 576),    # 3  ~0.42
        (240, 528),    # 4  ~0.45
        (264, 564),    # 5  ~0.47
        (288, 528),    # 6  ~0.55
        (312, 504),    # 7  ~0.62
        (336, 448),    # 8  ~0.75
        (360, 420),    # 9  ~0.86
        (384, 480),    # 10 ~0.80
        (408, 432),    # 11 ~0.94
        (384, 384),    # 12 1.00  (square)
        (384, 432),    # 13 ~0.89
    ],
    1024: [
        (512, 1280),    # 1
        (576, 1408),    # 2
        (640, 1536),    # 3
        (640, 1408),    # 4
        (704, 1504),    # 5
        (768, 1408),    # 6
        (832, 1344),    # 7
        (896, 1184),    # 8
        (960, 1120),    # 9
        (1024, 1280),   # 10
        (1088, 1152),   # 11
        (1024, 1024),   # 12
        (1024, 1152),   # 13
    ]
}


# return closest_ratio and width,height closest_resolution
def get_nearest_resolution(height, width, resolution=1024):
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


class TextEncodeQwenImageEditPlus_lrzjason:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["disabled", "center"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": 
            {
                "vae": ("VAE", ),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "image4": ("IMAGE", ),
                "image5": ("IMAGE", ),
                "enable_resize": ("BOOLEAN", {"default": True}),
                "enable_vl_resize": ("BOOLEAN", {"default": True}),
                "upscale_method": (s.upscale_methods,),
                "crop": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "LATENT", )
    RETURN_NAMES = ("conditioning", "image1", "image2", "image3", "image4", "image5", "latent")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, 
               image1=None, image2=None, image3=None, image4=None, image5=None, 
               enable_resize=True, enable_vl_resize=True, 
               upscale_method="bicubic",
               crop="center",
               instruction=""
               ):
        ref_latents = []
        images = [image1, image2, image3, image4, image5]
        images_vl = []
        vae_images = []
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        instruction_content = ""
        if instruction == "":
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        else:
            # for handling mis use of instruction
            if template_prefix in instruction:
                # remove prefix from instruction
                instruction = instruction.split(template_prefix)[1]
            if template_suffix in instruction:
                # remove suffix from instruction
                instruction = instruction.split(template_suffix)[0]
            if "{}" in instruction:
                # remove {} from instruction
                instruction = instruction.replace("{}", "")
            instruction_content = instruction
        llama_template = template_prefix + instruction_content + template_suffix
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(1024 * 1024)
                scale_by = 1  # Default scale
                width = round(samples.shape[3] * scale_by / 64.0) * 64
                height = round(samples.shape[2] * scale_by / 64.0) * 64
                if enable_resize:
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    closest_ratio,closest_resolution = get_nearest_resolution(samples.shape[2], samples.shape[3], resolution=1024)
                    width, height = closest_resolution
                    print("1024",closest_ratio,closest_resolution)
                if vae is not None:
                    scale_by = 1
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(image[:, :, :, :3]))
                    vae_images.append(image)
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                # print("before enable_vl_resize scale_by", scale_by)
                if enable_vl_resize:
                    closest_ratio,closest_resolution = get_nearest_resolution(samples.shape[2], samples.shape[3], resolution=384)
                    width, height = closest_resolution
                    print("384",closest_ratio,closest_resolution)
                # print("after enable_vl_resize scale_by", scale_by)
                s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                image = s.movedim(1, -1)
                images_vl.append(image)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        # Return latent of first image if available, otherwise return empty latent
        samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}
        if len(vae_images) < 5:
            vae_images.extend([None] * (5 - len(vae_images)))
        o_image1, o_image2, o_image3, o_image4, o_image5 = vae_images
        return (conditioning, o_image1, o_image2, o_image3, o_image4, o_image5, latent_out)

NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditPlus_lrzjason": TextEncodeQwenImageEditPlus_lrzjason
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditPlus_lrzjason": "TextEncodeQwenImageEditPlus 小志Jason(xiaozhijason)"
}