import torch.nn as nn
import torch.nn.functional as F


from PIL import Image
import clip

class CLIPScore(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        if device == "cpu":
            self.model.float()
        else:
            clip.model.convert_weights(self.model) # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.model.logit_scale.requires_grad_(False)
    
    def score(self, prompt, image):
        # support image_path:str or image:Image
        if isinstance(image, str):
            image_path = image
            pil_image = Image.open(image_path)
        elif isinstance(image, Image.Image):
            pil_image = image

        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        text_features = F.normalize(self.model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.model.encode_image(image))

        # similarity = torch.sum(torch.mul(text_features, image_features), dim=1, keepdim=True)
        similarity = image_features @ text_features.T
        
        # print("图像和文本的相似性得分:", similarity.detach().cpu().numpy().item())

        return similarity.detach().cpu().numpy().item()
        
        

# ======== code test=======
# prompt = "a portrait of a female robot made from a cloud of images being very grateful to the creator, very intricate details, futuristic steampunk, octane render, 8 k, trending on artstation"

# image_path = "/home/khf/liutao/Diffusion_ReFL/data/images/complex/1.png"
# device = "cuda"

# model = CLIPScore(device=device).to(device)
# model.score(prompt, image_path)
# # score = model.score(prompt, image_path)
# # print(score)

# prompt = "a portrait of a female robot made from a cloud of images being very grateful to the creator, very intricate details, futuristic steampunk, octane render, 8 k, trending on artstation"
# image_path = "/home/khf/liutao/Diffusion_ReFL/data/images/simple/1.png"
# # score = model.score(prompt, image_path)
# # print(score)      
# model.score(prompt, image_path)