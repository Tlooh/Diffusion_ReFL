import torch
import torch.nn as nn

from PIL import Image
from models.BLIP.blip_pretrain import BLIP_Pretrain

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)


class MyReward(nn.Module):
    def __init__(self, med_config, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.blip_model = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        print("1.load BLIP model")
        self.preprocess = _transform(224)
        self.mlp = MLP(768)
        
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072
    

    def score_gard(self, prompt_ids, prompt_attention_mask, image):
        
        image_embeds = self.blip_model.visual_encoder(image)
        
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip_model.text_encoder(prompt_ids,
                                                    attention_mask = prompt_attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                )
        
        txt_features = text_output.last_hidden_state[:,0,:] # (feature_dim)
        
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        
        return rewards



# device = "cuda"
# med_config = "/home/khf/liutao/Diffusion_ReFL/configs/bert_config.json"

# # blip_model_download_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth"

# reward_model = MyReward(med_config=med_config, device=device).to(device)

# complex_prompt = "a portrait of a female robot made from a cloud of images being very grateful to the creator, very intricate details, futuristic steampunk, octane render, 8 k, trending on artstation"

# complex_image = "/home/khf/liutao/data/images/complex/0.png"

# from diffusers import StableDiffusionPipeline

# model = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path="/home/khf/liutao/sd1-4")
# model.to(device)
# tokenizer = model.tokenizer
# text_encoder = model.text_encoder

# complex_image = model(prompt=complex_prompt, output_type="pt").images
# # text_inputs = tokenizer(complex_prompt, max_length = tokenizer.model_max_length, padding="max_length", return_tensors="pt")
# # print(text_inputs)

# text_inputs = tokenizer(complex_prompt, max_length = 35,  padding="max_length", return_tensors="pt")
# prompt_ids = text_inputs.input_ids.to(device)
# prompt_attention_mask = text_inputs.attention_mask.to(device)

# print(prompt_ids)
# print(prompt_attention_mask)
# # print(complex_image)


# complex_image = (complex_image / 2 + 0.5).clamp(0, 1)
# print(complex_image.shape)

# # image encode
# def _transform():
#     return Compose([
#         Resize(224, interpolation=BICUBIC),
#         CenterCrop(224),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

# rm_preprocess = _transform()
# # complex_image = rm_preprocess(complex_image).detach().cpu()
# complex_image = rm_preprocess(complex_image).to(device)
# print(f"complex image shape: {complex_image.shape}, device: {complex_image.device}")
# reward_model.score_gard(prompt_ids, prompt_attention_mask, complex_image)

# download_root = None
# model_download_root = download_root or os.path.expanduser("~/.cache/ImageReward")
# print(model_download_root)