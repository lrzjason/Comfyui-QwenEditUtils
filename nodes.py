import node_helpers
import comfy.utils
import math
import torch
import torch
import numpy as np
from PIL import Image
import json
import os
import copy
import folder_paths
import hashlib

from PIL import Image, ImageOps, ImageSequence

class CropWithPadInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pad_info": ("ANY", ),  # pad_info dictionary containing x, y, width, height and scale
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("cropped_image", "scale_by", )
    FUNCTION = "crop_image"

    CATEGORY = "image"

    def crop_image(self, image, pad_info):
        # Extract pad information from the original padding process:
        # In the original function:
        # - resized_samples are placed at canvas[:, :, :resized_height, :resized_width]
        # - pad_info = {"x": 0, "y": 0, "width": canvas_width - resized_width, "height": canvas_height - resized_height}
        x = pad_info.get("x", 0)  # This is always 0 in the original function
        y = pad_info.get("y", 0)  # This is always 0 in the original function
        width_padding = pad_info.get("width", 0)  # Right/bottom padding added
        height_padding = pad_info.get("height", 0)  # Right/bottom padding added
        scale_by = pad_info.get("scale_by", 1.0)
        
        img = image.movedim(-1, 1)  # Convert from (H, W, C) to (C, H, W)
        
        # Calculate the original content dimensions before padding was added
        original_content_width = img.shape[3] - width_padding
        original_content_height = img.shape[2] - height_padding
        
        # Crop to get just the original content area (which was placed at position (0,0))
        cropped_img = img[:, :, x:original_content_height, y:original_content_width]
        
        # Convert back to (H, W, C) format
        cropped_image = cropped_img.movedim(1, -1)
        
        return (cropped_image, scale_by)


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
                
            images = [image]
            if vae is not None:
                ref_latent = vae.encode(image)
                # print("ref_latent:",ref_latent.shape)
        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]})
            
        return (conditioning, image, {"samples":ref_latent}, )

def get_system_prompt(instruction):
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
    
    return llama_template

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
                "skip_first_image_resize": ("BOOLEAN", {"default": False}),
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
               enable_resize=True, enable_vl_resize=True, skip_first_image_resize=False,
               upscale_method="bicubic",
               crop="center",
               instruction=""
               ):
        ref_latents = []
        images = [image1, image2, image3, image4, image5]
        images_vl = []
        vae_images = []
        # template_prefix = "<|im_start|>system\n"
        # template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # instruction_content = ""
        # if instruction == "":
        #     instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        # else:
        #     # for handling mis use of instruction
        #     if template_prefix in instruction:
        #         # remove prefix from instruction
        #         instruction = instruction.split(template_prefix)[1]
        #     if template_suffix in instruction:
        #         # remove suffix from instruction
        #         instruction = instruction.split(template_suffix)[0]
        #     if "{}" in instruction:
        #         # remove {} from instruction
        #         instruction = instruction.replace("{}", "")
        #     instruction_content = instruction
        # llama_template = template_prefix + instruction_content + template_suffix
        llama_template = get_system_prompt(instruction)
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                current_total = (samples.shape[3] * samples.shape[2])
                total = int(1024 * 1024)
                scale_by = 1  # Default scale
                if enable_resize:
                    scale_by = math.sqrt(total / current_total)
                width = round(samples.shape[3] * scale_by / 8.0) * 8
                height = round(samples.shape[2] * scale_by / 8.0) * 8
                if vae is not None:
                    s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(image[:, :, :, :3]))
                    vae_images.append(image)
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                # print("before enable_vl_resize scale_by", scale_by)
                # print("before enable_vl_resize width,height", width,height)
                if enable_vl_resize and not skip_first_image_resize and i == 0:
                    total = int(384 * 384)
                    scale_by = math.sqrt(total / current_total)
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                # print("after enable_vl_resize width,height", width,height)
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



class TextEncodeQwenImageEditPlusAdvance_lrzjason:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [392,384]
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
                "vl_resize_image1": ("IMAGE", ),
                "vl_resize_image2": ("IMAGE", ),
                "vl_resize_image3": ("IMAGE", ),
                "not_resize_image1": ("IMAGE", ),
                "not_resize_image2": ("IMAGE", ),
                "not_resize_image3": ("IMAGE", ),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": (s.target_vl_sizes, {"default": 384}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "ANY")
    RETURN_NAMES = ("conditioning_with_full_ref", "latent", "target_image1", "target_image2", "target_image3", "vl_resized_image1", "vl_resized_image2", "vl_resized_image3", "conditioning_with_first_ref", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, 
               vl_resize_image1=None, vl_resize_image2=None, vl_resize_image3=None,
               not_resize_image1=None, not_resize_image2=None, not_resize_image3=None, 
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="center",
               instruction="",
               ):
        
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0
        }
        ref_latents = []
        images = [not_resize_image1, not_resize_image2, not_resize_image3, 
                  vl_resize_image1, vl_resize_image2, vl_resize_image3]
        vl_resized_images = []
        
        images = [
            {
                "image": not_resize_image1,
                "vl_resize": False 
            },
            {
                "image": not_resize_image2,
                "vl_resize": False 
            },
            {
                "image": not_resize_image3,
                "vl_resize": False 
            },
            {
                "image": vl_resize_image1,
                "vl_resize": True 
            },
            {
                "image": vl_resize_image2,
                "vl_resize": True 
            },
            {
                "image": vl_resize_image3,
                "vl_resize": True 
            }
        ]
        
        vae_images = []
        vl_images = []
        # template_prefix = "<|im_start|>system\n"
        # template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # instruction_content = ""
        # if instruction == "":
        #     instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        # else:
        #     # for handling mis use of instruction
        #     if template_prefix in instruction:
        #         # remove prefix from instruction
        #         instruction = instruction.split(template_prefix)[1]
        #     if template_suffix in instruction:
        #         # remove suffix from instruction
        #         instruction = instruction.split(template_suffix)[0]
        #     if "{}" in instruction:
        #         # remove {} from instruction
        #         instruction = instruction.replace("{}", "")
        #     instruction_content = instruction
        # llama_template = template_prefix + instruction_content + template_suffix
        llama_template = get_system_prompt(instruction)
        image_prompt = ""

        if vae is not None:
            for i, image_obj in enumerate(images):
                image = image_obj["image"]
                vl_resize = image_obj["vl_resize"]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(target_size * target_size)
                    scale_by = math.sqrt(total / current_total)
                    if crop_method == "pad":
                        crop = "center"
                        # pad image to upper size
                        scaled_width = round(samples.shape[3] * scale_by)
                        scaled_height = round(samples.shape[2] * scale_by)
                        canvas_width = math.ceil(samples.shape[3] * scale_by / 8.0) * 8
                        canvas_height = math.ceil(samples.shape[2] * scale_by / 8.0) * 8
                        
                        # pad image to canvas size
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        pad_info = {
                            "x": 0,
                            "y": 0,
                            "width": canvas_width - resized_width,
                            "height": canvas_height - resized_height,
                            "scale_by": 1 / scale_by
                        }
                        s = canvas
                    else:
                        width = round(samples.shape[3] * scale_by / 8.0) * 8
                        height = round(samples.shape[2] * scale_by / 8.0) * 8
                        crop = crop_method
                        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(image[:, :, :, :3]))
                    vae_images.append(image)
                    
                    if vl_resize:
                        # print("vl_resize")
                        total = int(target_vl_size * target_vl_size)
                        scale_by = math.sqrt(total / current_total)
                        
                        if crop_method == "pad":
                            crop = "center"
                            # pad image to upper size
                            scaled_width = round(samples.shape[3] * scale_by)
                            scaled_height = round(samples.shape[2] * scale_by)
                            canvas_width = math.ceil(samples.shape[3] * scale_by)
                            canvas_height = math.ceil(samples.shape[2] * scale_by)
                            
                            # pad image to canvas size
                            canvas = torch.zeros(
                                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                            resized_width = resized_samples.shape[3]
                            resized_height = resized_samples.shape[2]
                            
                            canvas[:, :, :resized_height, :resized_width] = resized_samples
                            s = canvas
                        else:
                            width = round(samples.shape[3] * scale_by)
                            height = round(samples.shape[2] * scale_by)
                            crop = crop_method
                            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        image = s.movedim(1, -1)
                        vl_resized_images.append(image)
                    # handle non resize vl images
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_full_ref = conditioning
        conditioning_with_first_ref = conditioning
        if len(ref_latents) > 0:
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            
            conditioning_with_first_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[0]]}, append=True)
        # Return latent of first image if available, otherwise return empty latent
        samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}
        if len(vae_images) < 3:
            vae_images.extend([None] * (3 - len(vae_images)))
        o_image1, o_image2, o_image3 = vae_images
        
        if len(vl_resized_images) < 3:
            vl_resized_images.extend([None] * (3 - len(vl_resized_images)))
        vl_image1, vl_image2, vl_image3 = vl_resized_images
        
        return (conditioning_full_ref, latent_out, o_image1, o_image2, o_image3, vl_image1, vl_image2, vl_image3, conditioning_with_first_ref, pad_info)

def validate_vl_resize_indexs(vl_resize_indexs_str, valid_length):
    try:
        indexes = [int(i)-1 for i in vl_resize_indexs_str.split(",")]
        # remove duplicates
        indexes = list(set(indexes))
    except ValueError as e:
        raise ValueError(f"Invalid format for vl_resize_indexs: {e}")

    if not indexes:
        raise ValueError("vl_resize_indexs must not be empty")

    indexes = [idx for idx in indexes if 0 <= idx < valid_length]

    return indexes

class TextEncodeQwenImageEditPlusPro_lrzjason:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [1024, 1344, 1536, 2048, 768, 512]
    target_vl_sizes = [392,384]
    vl_resize_indexs = [1,2,3]
    main_image_index = 1
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
                "vl_resize_indexs": ("STRING", {"default": "1,2,3"}),
                "main_image_index": ("INT", {"default": 1, "max": 5, "min": 1}),
                "target_size": (s.target_sizes, {"default": 1024}),
                "target_vl_size": (s.target_vl_sizes, {"default": 384}),
                "upscale_method": (s.upscale_methods,),
                "crop_method": (s.crop_methods,),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "ANY")
    RETURN_NAMES = ("conditioning_with_full_ref", "latent", "image1", "image2", "image3", "image4", "image5", "conditioning_with_main_ref", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"
    def encode(self, clip, prompt, vae=None, 
               image1=None, image2=None, image3=None,
               image4=None, image5=None, 
               vl_resize_indexs="1,2,3",
               main_image_index=1,
               target_size=1024, 
               target_vl_size=384,
               upscale_method="lanczos",
               crop_method="center",
               instruction="",
               ):
        # check vl_resize_indexs is valid indexes and not out of range
        resize_indexs = validate_vl_resize_indexs(vl_resize_indexs,5)
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 0
        }
        ref_latents = []
        temp = [image1, image2, image3, image4, image5]
        images = []
        for i, image in enumerate(temp):
            image_dict = {
                "image": image,
                "vl_resize": False
            }
            if i in resize_indexs:
                image_dict['vl_resize'] = True
            images.append(image_dict)
        vl_resized_images = []
        
        vae_images = []
        vl_images = []
        # template_prefix = "<|im_start|>system\n"
        # template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # instruction_content = ""
        # if instruction == "":
        #     instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        # else:
        #     # for handling mis use of instruction
        #     if template_prefix in instruction:
        #         # remove prefix from instruction
        #         instruction = instruction.split(template_prefix)[1]
        #     if template_suffix in instruction:
        #         # remove suffix from instruction
        #         instruction = instruction.split(template_suffix)[0]
        #     if "{}" in instruction:
        #         # remove {} from instruction
        #         instruction = instruction.replace("{}", "")
        #     instruction_content = instruction
        # llama_template = template_prefix + instruction_content + template_suffix
        llama_template = get_system_prompt(instruction)
        image_prompt = ""

        if vae is not None:
            for i, image_obj in enumerate(images):
                image = image_obj["image"]
                vl_resize = image_obj["vl_resize"]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(target_size * target_size)
                    scale_by = math.sqrt(total / current_total)
                    if crop_method == "pad":
                        crop = "center"
                        # pad image to upper size
                        scaled_width = round(samples.shape[3] * scale_by)
                        scaled_height = round(samples.shape[2] * scale_by)
                        canvas_width = math.ceil(samples.shape[3] * scale_by / 8.0) * 8
                        canvas_height = math.ceil(samples.shape[2] * scale_by / 8.0) * 8
                        
                        # pad image to canvas size
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        pad_info = {
                            "x": 0,
                            "y": 0,
                            "width": canvas_width - resized_width,
                            "height": canvas_height - resized_height,
                            "scale_by": 1 / scale_by
                        }
                        s = canvas
                    else:
                        width = round(samples.shape[3] * scale_by / 8.0) * 8
                        height = round(samples.shape[2] * scale_by / 8.0) * 8
                        crop = crop_method
                        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                    image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(image[:, :, :, :3]))
                    vae_images.append(image)
                    
                    if vl_resize:
                        # print("vl_resize")
                        total = int(target_vl_size * target_vl_size)
                        scale_by = math.sqrt(total / current_total)
                        
                        if crop_method == "pad":
                            crop = "center"
                            # pad image to upper size
                            scaled_width = round(samples.shape[3] * scale_by)
                            scaled_height = round(samples.shape[2] * scale_by)
                            canvas_width = math.ceil(samples.shape[3] * scale_by)
                            canvas_height = math.ceil(samples.shape[2] * scale_by)
                            
                            # pad image to canvas size
                            canvas = torch.zeros(
                                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                                dtype=samples.dtype,
                                device=samples.device
                            )
                            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
                            resized_width = resized_samples.shape[3]
                            resized_height = resized_samples.shape[2]
                            
                            canvas[:, :, :resized_height, :resized_width] = resized_samples
                            s = canvas
                        else:
                            width = round(samples.shape[3] * scale_by)
                            height = round(samples.shape[2] * scale_by)
                            crop = crop_method
                            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
                        
                        image = s.movedim(1, -1)
                        vl_resized_images.append(image)
                    # handle non resize vl images
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                    vl_images.append(image)
                    
                
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning_full_ref = conditioning
        conditioning_with_main_ref = conditioning
        samples = torch.zeros(1, 4, 128, 128)
        
        if len(ref_latents) > 0:
            # remap main_image_index from start from 1 to 0
            main_image_index = main_image_index - 1
            if main_image_index >= len(ref_latents):
                print("\n Auto fixing main_image_index to the first image index")
                main_image_index = 0
            conditioning_full_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            conditioning_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_index]]}, append=True)
            # Return latent of first image if available, otherwise return empty latent
            # samples = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
            samples = ref_latents[main_image_index]
        latent_out = {"samples": samples}
        if len(vae_images) < len(images):
            vae_images.extend([None] * (len(images) - len(vae_images)))
        image1, image2, image3, image4, image5 = vae_images
        
        return (conditioning_full_ref, latent_out, image1, image2, image3, image4, image5, conditioning_with_main_ref, pad_info)


class TextEncodeQwenImageEditPlusCustom_lrzjason:
    # upscale_methods = ["lanczos", "bicubic", "area"]
    # crop_methods = ["pad", "center", "disabled"]
    # example_config = {
    #     "image": None,
    #     # ref part
    #     "to_ref": True,
    #     "ref_main_image": True,
    #     "ref_longest_edge": 1024,
    #     "ref_crop": "center", #"pad" for main image, "center", "disabled"
    #     "ref_upscale": "lanczos",
        
    #     # vl part
    #     "to_vl": True,
    #     "vl_resize": True,
    #     "vl_target_size": 384,
    #     "vl_crop": "center",
    #     "vl_upscale": "bicubic", #to scale image down, "bicubic", "area" might better than "lanczos"
    # }
    # example_output = {
    #     "pad_info": pad_info,
    #     "noise_mask": noise_mask,
    #     "full_refs_cond": conditioning,
    #     "main_ref_cond": conditioning_only_with_main_ref,
    #     "main_image": main_image,
    #     "vae_images": vae_images,
    #     "ref_latents": ref_latents,
    #     "vl_images": vl_images,
    #     "full_prompt": full_prompt,
    #     "llama_template": llama_template
    # }
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "configs": ("LIST", {"default": None})
            },
            "optional": 
            {
                "return_full_refs_cond": ("BOOLEAN", {"default": True}),
                # "set_noise_mask": ("BOOLEAN", {"default": False, "tooltip": "Only useful when using ref_crop == pad. It would automatically mask out the padding area."}),
                "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),   
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY")
    RETURN_NAMES = ("conditioning", "latent", "custom_output")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"
    def encode(self, clip, vae, prompt, 
               configs=None,
               return_full_refs_cond=True,
            #    set_noise_mask=False,
               instruction="",
        ):
        # print("len(configs)")
        # print(len(configs))
        llama_template = get_system_prompt(instruction)
        image_prompt = ""
        
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 1.0,
        }
        # print("len(configs)", len(configs))
        # check len(configs)
        assert len(configs) > 0, "No image provided"
        
        main_image_index = -1
        for i, image_obj in enumerate(configs):
            if image_obj["to_ref"]:
                if main_image_index == -1 and image_obj["ref_main_image"]:
                    main_image_index = i
                    continue
                # ensure only one main image
                if main_image_index != -1:
                    image_obj["ref_main_image"] = False
        if main_image_index == -1:
            print("\n Auto fixing main_image_index to the first image index")
            main_image_index = 0
        
        ref_latents = []
        vae_images = []
        vl_images = []
        
        noise_mask = None
        for i, image_obj in enumerate(configs):
            assert "image" in image_obj, "Image is missing"
            image = image_obj["image"]
            to_ref = image_obj["to_ref"]
            ref_main_image = image_obj["ref_main_image"]
            ref_longest_edge = image_obj["ref_longest_edge"]
            ref_crop = image_obj["ref_crop"]
            ref_upscale = image_obj["ref_upscale"]
            
            to_vl = image_obj["to_vl"]
            vl_resize = image_obj["vl_resize"]
            vl_target_size = image_obj["vl_target_size"]
            vl_crop = image_obj["vl_crop"]
            vl_upscale = image_obj["vl_upscale"]
            
            mask = None
            if "mask" in image_obj:
                mask = image_obj["mask"]
            
            samples = image.movedim(-1, 1)
            if mask is not None:
                _, c, _, _ = samples.shape
                sample_masks = mask.unsqueeze(1).repeat(1, c, 1, 1)  # same shape
                # sample_masks = mask.movedim(-1, 1)
                # check samples and sample_masks should match
                print('samples.shape',samples.shape)
                print('sample_masks.shape',sample_masks.shape)
                assert samples.shape == sample_masks.shape, "Image and mask shape mismatch"
            
            if not to_ref and not to_vl:
                continue
            if to_ref:
                # print("ori_image.shape",samples.shape)
                # ori_height, ori_width = samples.shape[2:]
                ori_longest_edge = max(samples.shape[2], samples.shape[3])
                scale_by = ori_longest_edge / ref_longest_edge
                scaled_height = int(round(samples.shape[2] / scale_by))
                scaled_width = int(round(samples.shape[3] / scale_by))
                
                # pad only apply to main image
                if ref_crop == "pad" and ref_main_image:
                    # print("In pad mode")
                    # print("scaled_width", scaled_width)
                    # print("scaled_height", scaled_height)
                    crop = "center"
                    canvas_width = math.ceil(scaled_width / 8.0) * 8
                    canvas_height = math.ceil(scaled_height / 8.0) * 8
                    # pad image to canvas size
                    canvas = torch.zeros(
                        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                        dtype=samples.dtype,
                        device=samples.device
                    )
                    
                    resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, ref_upscale, crop)
                    
                    resized_width = resized_samples.shape[3]
                    resized_height = resized_samples.shape[2]
                    
                    # set resized samples to canvas
                    # canvas[:, :, x_offset:resized_height, y_offset:resized_width] = resized_samples
                    canvas[:, :, :resized_height, :resized_width] = resized_samples
                    
                    
                    # if set_noise_mask:
                        # noise_mask = torch.zeros(canvas.shape, dtype=torch.bool, device=canvas.device)
                        # noise_mask[:, :, x_offset:resized_height, y_offset:resized_width] = 1.0
                        # print("noise_mask.shape", noise_mask.shape)
                        # noise_mask = noise_mask.movedim(1, -1)
                        # print("movedim noise_mask.shape", noise_mask.shape)
                    
                    # only return main image pad info
                    
                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(resized_width * resized_height)
                    scale_by = math.sqrt(total / current_total)
                    pad_info = {
                        "x": 0,
                        "y": 0,
                        "width": canvas_width - resized_width,
                        "height": canvas_height - resized_height,
                        "scale_by": round(1 / scale_by, 3)
                    }
                    
                    # print("pad_info", pad_info)
                    s = canvas
                    
                    if mask is not None and ref_main_image:
                        mask_canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        
                        resized_sample_masks = comfy.utils.common_upscale(sample_masks, scaled_width, scaled_height, ref_upscale, crop)
                        mask_canvas[:, :, :resized_height, :resized_width] = resized_sample_masks
                        m =  mask_canvas
                        
                        # remove noise mask channel
                        noise_mask = m[:, :1, :, :].squeeze(1)
                        print("noise_mask.shape", noise_mask.shape)
                else:
                    crop = ref_crop
                    # handle pad method when not main image
                    if ref_crop == "pad":
                        crop = "center"
                    width = round(scaled_width / 8.0) * 8
                    height = round(scaled_height / 8.0) * 8
                    # print("width",width)
                    # print("height",height)
                    s = comfy.utils.common_upscale(samples, width, height, ref_upscale, crop)
                    
                    if mask is not None and ref_main_image:
                        m = comfy.utils.common_upscale(sample_masks, width, height, ref_upscale, crop)
                        # remove noise mask channel
                        noise_mask = m[:, :1, :, :].squeeze(1)
                        print("noise_mask.shape", noise_mask.shape)
                image = s.movedim(1, -1)
                ref_latents.append(vae.encode(image[:, :, :, :3]))
                vae_images.append(image)
            if to_vl:
                if vl_resize:
                    # print("vl_resize")
                    total = int(vl_target_size * vl_target_size)
                else:
                    total = int(samples.shape[3] * samples.shape[2])
                    if total > 2048 * 2048:
                        print("vl_target_size too large, clipping to 2048")
                        total = 2048 * 2048
                current_total = (samples.shape[3] * samples.shape[2])
                scale_by = math.sqrt(total / current_total)
            
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
                
                image = s.movedim(1, -1)
                # handle non resize vl images
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                vl_images.append(image)
                

        full_prompt = image_prompt + prompt
        tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        samples = torch.zeros(1, 4, 128, 128)
        conditioning_only_with_main_ref = None
        if len(ref_latents) > 0:
            conditioning_only_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_index]]}, append=True)
            conditioning_full_refs = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            samples = ref_latents[main_image_index]
        latent_out = {"samples": samples}
        
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
        
        conditioning_output = conditioning_full_refs
        if not return_full_refs_cond:
            conditioning_output = conditioning_only_with_main_ref
        
        main_image = None
        if len(vae_images)>0:
            main_image = vae_images[main_image_index]
        
        custom_output = {
            "pad_info": pad_info,
            "full_refs_cond": conditioning_full_refs,
            "main_ref_cond": conditioning_only_with_main_ref,
            "main_image": main_image,
            "vae_images": vae_images,
            "ref_latents": ref_latents,
            "vl_images": vl_images,
            "full_prompt": full_prompt,
            "llama_template": llama_template,
            "no_refs_cond": conditioning,
            "mask": noise_mask,
        }
        
        return (conditioning_output, latent_out, custom_output)


class QwenEditConfigJsonParser():
    default_config = {
        "to_ref": True,
        "ref_main_image": False,
        "ref_longest_edge": 1024,
        "ref_crop": "center",
        "ref_upscale": "lanczos",
        "to_vl": True,
        "vl_resize": True,
        "vl_target_size": 384,
        "vl_crop": "center",
        "vl_upscale": "bicubic",
        "mask": None
    }
    
    default_config_json = """{
    "to_ref": true,
    "ref_main_image": false,
    "ref_longest_edge": 1024,
    "ref_crop": "center",
    "ref_upscale": "lanczos",
    "to_vl": true,
    "vl_resize": true,
    "vl_target_size": 384,
    "vl_crop": "center",
    "vl_upscale": "bicubic"
}"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "image": ("IMAGE", ),
            },
            "optional": 
            {
                "configs": ("LIST", {"default": None, "tooltip": "Configs list"}),
                "config_json": ("STRING", {"default": s.default_config_json, "multiline": True, "tooltip": "Config JSON String"}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"
    CATEGORY = "advanced/conditioning"
    def prepare_config(self, image, configs=None,
                config_json="", mask=None
        ):
        if configs is None:
            configs = []
        # print("len(configs)", len(configs))
        
        config = self.default_config.copy()
        try:
            json_config = json.loads(config_json)
        except Exception as e:
            print(f"An error occurred while loading json_config")
            print(json_config)
        
        config.update(json_config)
        config["image"] = image
        
        config["mask"] = None
        
        # print("mask.shape", mask.shape)
        # print("image.shape", image.shape)
        if mask is not None:
            # check mask height,width equals image height,width
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                print("mask height,width not equals image height,width, skipping mask")
                mask = None
            config["mask"] = mask
        
        config_output = copy.deepcopy(configs)
        del configs
        
        config_output.append(config)
        # print("len(configs)", len(configs))
        return (config_output, config, )

class QwenEditConfigPreparer:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    vl_crop_methods = ["center", "disabled"]
    
    # example_config = {
    #     "image": None,
    #     # ref part
    #     "to_ref": True,
    #     "ref_main_image": True,
    #     "ref_longest_edge": 1024,
    #     "ref_crop": "center", #"pad" for main image, "center", "disabled"
    #     "ref_upscale": "lanczos",
        
    #     # vl part
    #     "to_vl": True,
    #     "vl_resize": True,
    #     "vl_target_size": 384,
    #     "vl_crop": "center",
    #     "vl_upscale": "bicubic", #to scale image down, "bicubic", "area" might better than "lanczos"
    # }   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "image": ("IMAGE", ),
            },
            "optional": 
            {
                "configs": ("LIST", {"default": None, "tooltip": "Configs list"}),
                "to_ref": ("BOOLEAN", {"default": True, "tooltip": "Add image to reference latent"}),
                "ref_main_image": ("BOOLEAN", {"default": True, "tooltip": "Set image as main image which would return the latent as output."}),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 1, "tooltip": "Longest edge of the output latent"}),
                "ref_crop": (s.crop_methods, {"default": "pad", "tooltip": "Crop method for reference image"}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for reference image"}),
    
                "to_vl": ("BOOLEAN", {"default": True, "tooltip": "Add image to qwenvl 2.5 encode"}),
                "vl_resize": ("BOOLEAN", {"default": True, "tooltip": "Resize image before qwenvl 2.5 encode"}),
                "vl_target_size": ("INT", {"default": 384, "min": 384, "max": 2048, "tooltip": "Target size of the qwenvl 2.5 encode"}),
                "vl_crop": (s.vl_crop_methods, {"default": "center", "tooltip": "Crop method for reference image"}),
                "vl_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for reference image"}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"
    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="center", ref_upscale="lanczos",
                to_vl=True, vl_resize=True, vl_target_size=384, vl_crop="center", vl_upscale="bicubic",
                mask=None
        ):
        if configs is None:
            configs = []
        # print("len(configs)", len(configs))
        # print("configs")
        # print(configs)
        config = {
            "image": image,
            "to_ref": to_ref,
            "ref_main_image": ref_main_image,
            "ref_longest_edge": ref_longest_edge,
            "ref_crop": ref_crop,
            "ref_upscale": ref_upscale,
            
            "to_vl": to_vl,
            "vl_resize": vl_resize,
            "vl_target_size": vl_target_size,
            "vl_crop": vl_crop,
            "vl_upscale": vl_upscale
        }
        
        
        config_output = copy.deepcopy(configs)
        
        # print("mask.shape", mask.shape)
        # print("image.shape", image.shape)
        if mask is not None:
            # check mask height,width equals image height,width
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                print("mask height,width not equals image height,width, skipping mask")
                mask = None
            config["mask"] = mask
        
        del configs
        
        
        config_output.append(config)
        # print("len(configs)", len(configs))
        return (config_output, config, )

class QwenEditOutputExtractor:
    preset_keys = [
        "pad_info",
        "full_refs_cond",
        "main_ref_cond",
        "main_image",
        "vae_images",
        "ref_latents",
        "vl_images",
        "full_prompt",
        "llama_template",
        "no_refs_cond",
        "mask"
    ]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "custom_output": ("ANY", ),
            }
        }

    # RETURN_TYPES = ("ANY", "MASK", "CONDITIONING", "CONDITIONING", "IMAGE", "LIST", "LIST", "LIST", "STRING", "STRING")
    # RETURN_NAMES = ("pad_info", "noise_mask", "full_refs_cond", "main_ref_cond", "main_image", "vae_images", "ref_latents", "vl_images", "full_prompt", "llama_template")
    
    RETURN_TYPES = ("ANY", "CONDITIONING", "CONDITIONING", "IMAGE", "LIST", "LIST", "LIST", "STRING", "STRING", "CONDITIONING", "MASK")
    RETURN_NAMES = ("pad_info", "full_refs_cond", "main_ref_cond", "main_image", "vae_images", "ref_latents", "vl_images", "full_prompt", "llama_template", "no_refs_cond", "mask")
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    
    def extract(self, custom_output):
        pad_info = custom_output['pad_info']
        # noise_mask = custom_output['noise_mask']
        full_refs_cond = custom_output['full_refs_cond']
        main_ref_cond = custom_output['main_ref_cond']
        main_image = custom_output['main_image']
        vae_images = custom_output['vae_images']
        ref_latents = custom_output['ref_latents']
        vl_images = custom_output['vl_images']
        full_prompt = custom_output['full_prompt']
        llama_template = custom_output['llama_template']
        no_refs_cond = custom_output['no_refs_cond']
        mask = custom_output['mask']
        
        return (pad_info, full_refs_cond, main_ref_cond, main_image, vae_images, ref_latents, vl_images, full_prompt, llama_template, no_refs_cond, mask)



class QwenEditListExtractor():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "items": ("LIST", ),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "Index of the image"}),
            }
        }

    RETURN_TYPES = ("ANY", )
    RETURN_NAMES = ("item", )
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    def extract(self, items, index):
        assert index < len(items), f"Index out of range, len(image_list): {len(items)}"
        
        return (items[index], )


class QwenEditAny2Image():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "item": ("ANY", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("item", )
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    def extract(self, item):
        return (item, )


class QwenEditAny2Latent():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "item": ("ANY", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("item", )
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    def extract(self, item):
        latent_out = {"samples": item}
        return (latent_out, )  

class QwenEditAdaptiveLongestEdge():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "image": ("IMAGE", ),
                "max_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 1, "tooltip": "When image is larger than max_size, it will be resized to under the max_size."}),
            },
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("longest_edge", )
    FUNCTION = "calculate_longest_edge"

    CATEGORY = "advanced/conditioning"
    def calculate_longest_edge(self, image, max_size):
        output = max(image.shape[1], image.shape[2])
        # print("image.shape[2], image.shape[3]", image.shape[1], image.shape[2])
        # print("longest_edge", output)
        if output <= max_size:
            return (output, )
        # Find how many times m fits into n
        k = int(math.ceil(output / max_size))
        # print("k", k)
        # Scale down by that factor
        
        output = int(output / k)
        # print("output", output)
        return (output, )  


class LoadImageReturnFilename:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", )
    RETURN_NAMES = ("image", "mask", "filename", )
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        # get filename from image_path without ext
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class ImageResizePlus:
    """独立的图像调整大小节点，提取自TextEncodeQwenImageEditPlusAdvance的图像处理逻辑"""
    
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    target_sizes = [512, 768, 1024, 1344, 1536, 2048]
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "target_size": (s.target_sizes, {"default": 1024}),
                "upscale_method": (s.upscale_methods, {"default": "lanczos"}),
                "crop_method": (s.crop_methods, {"default": "center"}),
            },
            "optional": {
                "target_vl_size": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 1}),
                "enable_vl_resize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "ANY")
    RETURN_NAMES = ("resized_image", "vl_resized_image", "pad_info")
    FUNCTION = "resize"

    CATEGORY = "image/processing"

    def resize(self, image, target_size=1024, upscale_method="lanczos", crop_method="center",
               target_vl_size=384, enable_vl_resize=False):
        """
        图像调整大小处理，提取自TextEncodeQwenImageEditPlusAdvance_lrzjason
        """
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 1.0
        }
        
        # 如果输入图像为None，返回空图像
        if image is None:
            # 创建一个空的图像张量
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, empty_image, pad_info)
        
        samples = image.movedim(-1, 1)
        current_total = (samples.shape[3] * samples.shape[2])
        total = int(target_size * target_size)
        scale_by = math.sqrt(total / current_total)
        
        if crop_method == "pad":
            crop = "center"
            # pad image to upper size
            scaled_width = round(samples.shape[3] * scale_by)
            scaled_height = round(samples.shape[2] * scale_by)
            canvas_width = math.ceil(samples.shape[3] * scale_by / 8.0) * 8
            canvas_height = math.ceil(samples.shape[2] * scale_by / 8.0) * 8
            
            # pad image to canvas size
            canvas = torch.zeros(
                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                dtype=samples.dtype,
                device=samples.device
            )
            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, upscale_method, crop)
            resized_width = resized_samples.shape[3]
            resized_height = resized_samples.shape[2]
            
            canvas[:, :, :resized_height, :resized_width] = resized_samples
            pad_info = {
                "x": 0,
                "y": 0,
                "width": canvas_width - resized_width,
                "height": canvas_height - resized_height,
                "scale_by": 1 / scale_by
            }
            s = canvas
        else:
            width = round(samples.shape[3] * scale_by / 8.0) * 8
            height = round(samples.shape[2] * scale_by / 8.0) * 8
            crop = crop_method
            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        
        resized_image = s.movedim(1, -1)
        
        # VL resize处理
        vl_resized_image = None
        if enable_vl_resize:
            total_vl = int(target_vl_size * target_vl_size)
            scale_by_vl = math.sqrt(total_vl / current_total)
            
            if crop_method == "pad":
                crop_vl = "center"
                scaled_width_vl = round(samples.shape[3] * scale_by_vl)
                scaled_height_vl = round(samples.shape[2] * scale_by_vl)
                canvas_width_vl = math.ceil(samples.shape[3] * scale_by_vl)
                canvas_height_vl = math.ceil(samples.shape[2] * scale_by_vl)
                
                canvas_vl = torch.zeros(
                    (samples.shape[0], samples.shape[1], canvas_height_vl, canvas_width_vl),
                    dtype=samples.dtype,
                    device=samples.device
                )
                resized_samples_vl = comfy.utils.common_upscale(samples, scaled_width_vl, scaled_height_vl, upscale_method, crop_vl)
                resized_width_vl = resized_samples_vl.shape[3]
                resized_height_vl = resized_samples_vl.shape[2]
                
                canvas_vl[:, :, :resized_height_vl, :resized_width_vl] = resized_samples_vl
                s_vl = canvas_vl
            else:
                width_vl = round(samples.shape[3] * scale_by_vl)
                height_vl = round(samples.shape[2] * scale_by_vl)
                crop_vl = crop_method
                s_vl = comfy.utils.common_upscale(samples, width_vl, height_vl, upscale_method, crop_vl)
            
            vl_resized_image = s_vl.movedim(1, -1)
        else:
            # 如果未启用VL调整，返回与resized_image相同的图像
            vl_resized_image = resized_image.clone()
        
        return (resized_image, vl_resized_image, pad_info)


NODE_CLASS_MAPPINGS = {
    "CropWithPadInfo": CropWithPadInfo,
    "TextEncodeQwenImageEdit_lrzjason": TextEncodeQwenImageEdit_lrzjason,
    "TextEncodeQwenImageEditPlus_lrzjason": TextEncodeQwenImageEditPlus_lrzjason,
    "TextEncodeQwenImageEditPlusAdvance_lrzjason": TextEncodeQwenImageEditPlusAdvance_lrzjason,
    "TextEncodeQwenImageEditPlusPro_lrzjason": TextEncodeQwenImageEditPlusPro_lrzjason,
    "TextEncodeQwenImageEditPlusCustom_lrzjason": TextEncodeQwenImageEditPlusCustom_lrzjason,
    "QwenEditOutputExtractor": QwenEditOutputExtractor,
    "QwenEditConfigPreparer": QwenEditConfigPreparer,
    "QwenEditConfigJsonParser": QwenEditConfigJsonParser,
    "QwenEditListExtractor": QwenEditListExtractor,
    "QwenEditAny2Image": QwenEditAny2Image,
    "QwenEditAny2Latent": QwenEditAny2Latent,
    "QwenEditAdaptiveLongestEdge": QwenEditAdaptiveLongestEdge,
    "LoadImageReturnFilename": LoadImageReturnFilename,
    "ImageResizePlus": ImageResizePlus
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "CropWithPadInfo": "Crop With Pad Info",
    "TextEncodeQwenImageEdit_lrzjason": "TextEncodeQwenImageEdit lrzjason",
    "TextEncodeQwenImageEditPlus_lrzjason": "TextEncodeQwenImageEditPlus lrzjason",
    "TextEncodeQwenImageEditPlusAdvance_lrzjason": "TextEncodeQwenImageEditPlusAdvance lrzjason",
    "TextEncodeQwenImageEditPlusPro_lrzjason": "TextEncodeQwenImageEditPlusPro lrzjason",
    "TextEncodeQwenImageEditPlusCustom_lrzjason": "TextEncodeQwenImageEditPlusCustom lrzjason",
    "QwenEditOutputExtractor": "Qwen Edit Output Extractor",
    "QwenEditConfigPreparer": "Qwen Edit Config Preparer",
    "QwenEditConfigJsonParser": "Qwen Edit Config Json Parser",
    "QwenEditListExtractor": "Qwen Edit List Extractor",
    "QwenEditAny2Image": "Qwen Edit Any2Image",
    "QwenEditAny2Latent": "Qwen Edit Any2Latent",
    "QwenEditAdaptiveLongestEdge": "Qwen Edit Adaptive Longest Edge",
    "LoadImageReturnFilename": "Load Image Return Filename",
    "ImageResizePlus": "Image Resize Plus"
}
