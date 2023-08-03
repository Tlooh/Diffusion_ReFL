from PIL import Image
import clip
import torch
image_path = ["/home/khf/liutao/Diffusion_ReFL/data/images/complex/1.png",
              "/home/khf/liutao/Diffusion_ReFL/data/images/simple/1.png",]

text_input = "a portrait of a female robot made from a cloud of images being very grateful to the creator, very intricate details, futuristic steampunk, octane render, 8 k, trending on artstation"
image_path1 = '/home/khf/liutao/Diffusion_ReFL/data/images/complex/1.png'
image_path2 = '/home/khf/liutao/Diffusion_ReFL/data/images/simple/2.png'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载图片并进行预处理
pil_image1 = Image.open(image_path1)
pil_image2 = Image.open(image_path2)
image_input1 = preprocess(pil_image1).unsqueeze(0).to(device)
image_input2 = preprocess(pil_image2).unsqueeze(0).to(device)

# 将文本输入编码为特征向量
text_input_encoded = clip.tokenize([text_input]).to(device)
text_features = model.encode_text(text_input_encoded)

# 将图片输入编码为特征向量
with torch.no_grad():
    image_features1 = model.encode_image(image_input1)
    image_features2 = model.encode_image(image_input2)

# 计算两个图片对于同一个文本的相似性得分
similarity1 = (image_features1 @ text_features.T)
similarity2 = (image_features2 @ text_features.T)

print("第一个图片和文本的相似性得分:", similarity1.item())
print("第二个图片和文本的相似性得分:", similarity2.item())



# import torch 
# from diffusers import StableDiffusionPipeline

# model_path = "/home/khf/liutao/sd1-4"
# pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
# pipe.to("cuda")
# prompts = ["a portrait of a female robot made from a cloud of images being very grateful to the creator, very intricate details, futuristic steampunk, octane render, 8 k, trending on artstation","grateful female robot portrait with intricate steampunk details"]

# seed = 8888
# g = torch.Generator().manual_seed(seed)
# images = pipe(prompt=prompts, generator = g).images
# images[0].save("yoda-pokemon-1.png")
# images[1].save("yoda-pokemon-2.png")

# accelerate launch --mixed_precision="fp16"  train_sd.py \
#   --pretrained_model_name_or_path="/home/khf/liutao/sd1-4" \
#   --dataset_name="/home/khf/liutao/Diffusion_ReFL/data/pokemon-blip-captions" \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="sd-pokemon-model" 


# accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#   --dataset_name="/home/khf/liutao/Diffusion_ReFL/data/pokemon-blip-captions" \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="sd-pokemon-model" 