import json
import torch 
import argparse
import os

import random
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler


# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# 4. schedule
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

device = "cuda"

vae.to(device)
text_encoder.to(device)
unet.to(device)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)



class CustomDataset(Dataset):
    def __init__(self, file_path):
        # 读取json文件
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.samples = data


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取样本数据
        sample = self.samples[idx]

        # 提取复杂和简单字段
        sample_id = sample['id']
        complex_data = sample['complex']
        simple_data = sample['simple']
        

        return sample_id, complex_data, simple_data

# 自定义 collate_fn，将元组中的两个字符串拆解为两个列表
def custom_collate_fn(batch):
    ids, complex_batch, simple_batch = zip(*batch)
    return list(ids), list(complex_batch), list(simple_batch)



def run_inference(g, prompts, device, id):

    complex_prompt = prompts[0]
    simple_prompt = prompts[1]
    # 1. get input_ids
    complex_input_ids = tokenizer(
        complex_prompt, 
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    simple_input_ids = tokenizer(
        simple_prompt, 
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids

    # 2.get prompt embedding
    # Get the text embedding for conditioning
    complex_encoder_hidden_states = text_encoder(complex_input_ids.to(device))[0]
    simple_encoder_hidden_states = text_encoder(simple_input_ids.to(device))[0]
    
    cond_prompt_embeds = torch.cat((complex_encoder_hidden_states, simple_encoder_hidden_states),dim=0)

    # Get the text embedding for un-conditioning
    uncond_input = tokenizer(
        [""] * 2, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    negative_prompt_embeds = text_encoder(
        uncond_input.input_ids.to(device),
        uncond_input.attention_mask.to(device)
    )[0]
                
    text_embeddings = torch.cat([negative_prompt_embeds, cond_prompt_embeds])
    print(text_embeddings.shape)

    # 4. Prepare timesteps
    scheduler.set_timesteps(100, device=device)
    timesteps = scheduler.timesteps
    latents = torch.randn((2, 4, 64, 64), generator=g, device=device).to(device)

    num_warmup_steps = len(timesteps) - 50 * scheduler.order

   
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            )[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy() # (1, 512, 512, 3)
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    pil_images[0].save(f"/home/khf/liutao/data/outputs/{id}_complex.png")
    pil_images[1].save(f"/home/khf/liutao/data/outputs/{id}_simple.png")
    print(f"img {id}: save  successfully!")


    



def generate_images(args):
   
    # 2. load prompt dataset

    # 创建自定义数据集
    dataset = CustomDataset(args.prompt_json_path)
    
    # 创建dataloader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 使用 len 函数获取 dataloader 的长度
    dataloader_length = len(data_loader)

    print("Dataloader的长度:", dataloader_length,"\t batch_size:", args.batch_size)
    # # print(data_loader)

    # 3. load model
    

    g = torch.Generator(device=device).manual_seed(args.seed)

    model = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16)
    model.to(device)

    # 4. aplly  prompt pair (complex-simple) to generate images
    for ids, complex_batch, simple_batch in data_loader:
        # 在这里进行训练或其他操作
        for i in range(args.batch_size):
            # a. get prompt pair
            image_id = ids[i]
            complex_prompt = complex_batch[i]
            simple_prompt = simple_batch[i]
            prompts = [complex_prompt, simple_prompt]
            
            # print(prompts)
            # # b. generate images
            run_inference(g, prompts, device, image_id)

            # print(f"第 {image_id} 个 prompt生成完成!")

            # images[0].save(f"{complex_imgs_dir}/{image_id}.png")
            # images[1].save(f"{simple_imgs_dir}/{image_id}.png")
            




if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/khf/liutao/sd1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=8888, help="A seed for reproducible inferencing.")
    parser.add_argument(
        "--prompt_json_path",
        default="/home/khf/liutao/data/prompts/prompts.json",
        type=str,
        help="Path to the prompts json list file, each item of which is a dict with keys `id` and `prompt`.",
    )
    parser.add_argument(
        "--img_dir",
        default="/home/khf/liutao/data/outputs",
        type=str,
        help="Path to the generated images directory. The sub-level directory name should be the name of the model and should correspond to the name specified in `model`.",
    )
    parser.add_argument(
        "--model",
        default="default",
        type=str,
        help="""default(["sd1-4", "sd2-1"]), all or any specified model names splitted with comma(,).""",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size (per device) for the prompt dataloader.",
    )
    parser.add_argument(
        "--gpu_id",
        default=None,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )

    args = parser.parse_args()
    
    generate_images(args)
            

