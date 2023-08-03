import torch
from typing import Any, Union, List

from metrics.CLIPScore import CLIPScore

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
    
    else:
        raise RuntimeError(f"Score {name} not found; available scores = {available_scores()}")
    
    print("checkpoint loaded")
    model.eval()

    return model
    