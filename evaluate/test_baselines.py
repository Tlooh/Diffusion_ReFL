import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from utils import load_score


def get_sorted_images(img_dir):
    # 获取目录下所有文件名（图片名）
    file_names = os.listdir(img_dir)

    # 排序文件名，以保证按数字顺序读取
    sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

    return sorted_file_names

def test_benchmark(args):
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")
    
    # load prompt samples and images_samples
    with open(args.prompts_path, "r") as f:
        prompts_data = json.load(f)
    

    # benchmarks list
    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]


    # resolve the generation model list to be evaluated
    model_names = []
    if args.model == "default":
        model_names = ["sd1-4"]

    else:
        model_names = args.model.split(",")
        model_names = [x.strip() for x in model_names]
    


    # 1. begin to evaluate with different metrics
    for benchmark_type in benchmark_types:
        # load the model for benchmark
        print(f"Loading {benchmark_type} model...")
        model = load_score(name=benchmark_type, device=device)
        print(f"{benchmark_type} benchmark begins!")

        # evaluate the model(s)
        with torch.no_grad():
            for model_name in model_names:
                print(f"model_names: {model_name}")
                # a. caculate CLIP score 
                for item in tqdm(prompts_data):
                    # load each prompt (comple / simple)
                    sample_id = item['id']
                    complex_prompt = item['complex']
                    simple_prompt = item['simple']

                    # load each image_path (comple / simple)
                    complex_image_path = os.path.join(args.img_dir, 'complex', f"{sample_id}.png")
                    simple_image_path = os.path.join(args.img_dir, 'simple', f"{sample_id}.png")

                    complex_clip_score = model.score(complex_prompt, complex_image_path)
                    simple_clip_score = model.score(simple_prompt, simple_image_path)

                    print(f"ID: {sample_id}")
                    print(f"Complex Image-Text similarity: {complex_clip_score}")
                    print(f"Simple Image-Text similarity: {simple_clip_score}")



                    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a script to test baselines.")

    parser.add_argument(
        "--prompts_path",
        default="/home/khf/liutao/data/prompts/prompts.json",
        type=str,
        help="Path to the prompts json list file, each item of which is a dict with keys `id` and `prompt`.",
    )
    parser.add_argument(
        "--img_dir",
        default="/home/khf/liutao/data/images",
        type=str,
        help="Path to the generated images directory. The sub-level directory name should be <complex> and <simple>",
    )
    parser.add_argument(
        "--model",
        default="default",
        type=str,
        help="""default(["sd1-4"]), all or any specified model names splitted with comma(,).""",
    )
    parser.add_argument(
        "--benchmark",
        default="CLIP",
        type=str,
        help="ImageReward-v1.0, Aesthetic, BLIP or CLIP, splitted with comma(,) if there are multiple benchmarks.",
    )
    parser.add_argument(
        "--gpu_id",
        default=None,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )

    args = parser.parse_args()

    test_benchmark(args)