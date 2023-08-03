import json
import torch 
import argparse
import os

import random


from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline

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

def run_inference(model, g, prompts):

    # a. generate images
    outputs = model(prompt=prompts, generator=g)
    has_nsfw = outputs.nsfw_content_detected

    if any(has_nsfw):
        random_seed = random.randint(1, 8888)
        g = torch.Generator().manual_seed(random_seed)
        print(f"replaced and set seed : {random_seed} and reproduce images")
        
        images = run_inference(model, g, prompts)

        return images
    
    images= outputs.images

    return images
        


def generate_images(args):
    # 1. generate dir to storage images
    if args.img_dir is not None:
        img_dir = os.path.expanduser(args.img_dir)
        print(os.path.expanduser(args.img_dir))
        complex_imgs_dir = os.path.join(img_dir, "complex")
        simple_imgs_dir = os.path.join(img_dir, "simple")

        if not os.path.exists(complex_imgs_dir):
            os.makedirs(complex_imgs_dir)

        if not os.path.exists(simple_imgs_dir):
            os.makedirs(simple_imgs_dir)
    
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
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")

    g = torch.Generator().manual_seed(args.seed)

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

            # b. generate images
            images = run_inference(model, g, prompts)

            print(f"第 {image_id} 个 prompt生成完成!")

            images[0].save(f"{complex_imgs_dir}/{image_id}.png")
            images[1].save(f"{simple_imgs_dir}/{image_id}.png")
            




if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/khf/liutao/sd1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inferencing.")
    parser.add_argument(
        "--prompt_json_path",
        default="/home/khf/liutao/Diffusion_ReFL/data/prompts/prompts.json",
        type=str,
        help="Path to the prompts json list file, each item of which is a dict with keys `id` and `prompt`.",
    )
    parser.add_argument(
        "--img_dir",
        default="/home/khf/liutao/Diffusion_ReFL/data/images",
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
            

