import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from utils.helper import set_device
from src.flux_manager import FluxManager

device = set_device()
prompt = "A night scene in Menton, Cote d'Azur, showing a midnight blue BMW E34 Touring car parked along a seaside promenade, with a backdrop of the sea. Scene includes palm trees and orange lamp lights."
#prompt = "Design a bougie asian girl sipping from a chic coffee cup. She should be sitting outside of a trendy café table, dressed casually yet stylishly in an oversized sweater, skinny jeans, and white Gucci sneakers. Use a soft, pastel color palette with hints of rose gold for her outfit and the coffee cup. Her hair should be in a high bun, with minimal makeup and a pair of Chanel sunglasses on. The vibe should be relaxed but still polished, like she’s enjoying a luxurious moment in her day., portrait photography, photo, fashion, vibrant"
total_num_imgs = 2
use_lora = True
use_img2img = False
use_controlnet = False
flux_manager = FluxManager()
flux_manager.set_model(model="schnell", quantize=8)
flux_manager.set_prompt(prompt=prompt)
flux_manager.set_output_layout(output="outputs/gen_img.png", width=256, height=256)
if use_lora:
    flux_manager.set_loras(
        lora_paths=[ "model/lora/F.1儿童简笔画风_v1.0.safetensors" ],
        lora_scales = [ 1.0 ],
        lora_triggers=[ "sketched style" ]
    )
if use_img2img: # cannot use img2img with controlnet
    flux_manager.set_img2img(
        init_image_path = "resource/img2img-refer-image.jpg",
        init_image_strength = 0.3
    )
if use_controlnet:
    flux_manager.set_controlnet(
        controlnet_image_path = "resource/controlnet-refer-image.png",
        controlnet_save_canny = True,
        controlnet_strength = 1.0,
    )
flux_manager.load_model(use_controlnet)
flux_manager.generate_imgs(num_imgs = total_num_imgs, use_controlnet = use_controlnet)
