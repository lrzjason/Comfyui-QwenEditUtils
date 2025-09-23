import node_helpers
import comfy.utils
import math
from PIL import Image
import numpy as np
import torch
import cv2
import copy


class TextEncodeQwenImageEditPlus_lrzjason:
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
                "llama_template": ("STRING", {"multiline": True, "default": "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "LATENT", )
    RETURN_NAMES = ("conditioning", "cropped_image", "latent")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None, image4=None, image5=None, enable_resize=True, resize_method="cv2", vl_encode_resize=False,llama_template=""):
        ref_latent = None
        images = [image1, image2, image3, image4, image5]
        images_vl = []
        vae_images = []
        ref_latents = []
        vae_image = None
        image_prompt = ""
        if llama_template is None or llama_template.strip() == "":
            llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        
        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    if enable_resize:
                        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                        vae_image = s.movedim(1, -1)[:, :, :, :3]
                    else:
                        vae_image = s.movedim(1, -1)[:, :, :, :3]
                   
                    vae_images.append(vae_image)
                    ref_latents.append(vae.encode(vae_image))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents is not None and len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents]})
            
        # Return latent of first image if available, otherwise return empty latent
        latent_out = {"samples": ref_latents[0]} if len(ref_latents) > 0 else {"samples": torch.zeros(1, 4, 128, 128)}
        
        return (conditioning, vae_images, latent_out)

NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditPlus_lrzjason": TextEncodeQwenImageEditPlus_lrzjason
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditPlus_lrzjason": "TextEncodeQwenImageEditPlus 小志Jason(xiaozhijason)"
}