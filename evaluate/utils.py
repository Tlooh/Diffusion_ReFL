import torch
import random
from typing import Any, Union, List

from metrics.CLIPScore import CLIPScore
from metrics.FIDScore import FIDScore
from diffusers import StableDiffusionPipeline

_SCORES = {
    "CLIP": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "BLIP": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
    "Aesthetic": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
}


def available_scores() -> List[str]:
    """Returns the names of available ImageReward scores"""
    return list(_SCORES.keys())


def load_score(name: str = "CLIP", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):


    if name == "CLIP":
        model = CLIPScore(device=device).to(device)
    
    elif name == "FID":
        print("FID model============")
        model = FIDScore(device=device, dims=2048).to(device)
    
    else:
        raise RuntimeError(f"Score {name} not found; available scores = {available_scores()}")
    
    print("checkpoint loaded")
    model.eval()

    return model


def load_sd_model(model_name: str = "sd1-4", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):

    if model_name == "sd1-4":
        model_id = "/home/khf/liutao/sd1-4"
    
    elif model_name == "sd2-1":
        model_id = "stabilityai/stable-diffusion-2-1"
    
    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)

    return model


def generate_image_pair(model, seed, prompts, output_type = "pil"):

    g = torch.Generator().manual_seed(seed)

    # a. generate images
    outputs = model(prompt=prompts, generator=g, output_type=output_type)
    has_nsfw = outputs.nsfw_content_detected

    if any(has_nsfw):
        random_seed = random.randint(1, 8888)
        print(f"reset seed : {random_seed} and reproduce images")
        
        images = generate_image_pair(model, random_seed, prompts)

        return images
    
    images= outputs.images

    return images


    