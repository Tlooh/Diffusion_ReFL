import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from utils import load_score, load_sd_model, generate_image_pair



def test_benchmark(args):
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")
    
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
    

    # load prompt samples and images_samples
    with open(args.prompts_path, "r") as f:
        prompts_data = json.load(f)
    num_pair_images = len(prompts_data)
    
    # begin to evaluate with different metrics
    for model_name in model_names:
        print(f"Begin evaluate model: {model_name}")
        model = load_sd_model(model_name=model_name, device=device)

        for benchmark_type in benchmark_types:
            # load the model for benchmark
            print(f"Loading {benchmark_type} model...")
            benchmark_metric = load_score(name=benchmark_type, device=device)
            print(f"{benchmark_type} benchmark begins!")

            # 1) evaluate SD with CLIP
            if benchmark_type == "CLIP":

                # load each prompt (comple / simple)
                for item in tqdm(prompts_data):
                
                    sample_id = item['id']
                    complex_prompt = item['complex']
                    simple_prompt = item['simple']

                    complex_image, simple_image = generate_image_pair(
                        model, 
                        seed=args.seed, 
                        prompts=[complex_prompt, simple_prompt])
                    
                    complex_clip_score = benchmark_metric.score(complex_prompt, complex_image)
                    simple_clip_score = benchmark_metric.score(simple_prompt, simple_image)
                    
                    print(f"ID: {sample_id}")
                    print(f"Complex Image-Text similarity: {complex_clip_score}")
                    print(f"Simple Image-Text similarity: {simple_clip_score}")

            # 2) prepare evaluate SD with FID
            elif benchmark_type == "FID":

                dims = 2048
                pred_act1 = np.empty((num_pair_images, dims))
                pred_act2 = np.empty((num_pair_images, dims))
                start_idx = 0
                complex_imgs_dir = "/home/khf/liutao/data/images/complex"
                simple_imgs_dir = "/home/khf/liutao/data/images/simple"

                # load each prompt (comple / simple)
                for item in tqdm(prompts_data):
                
                    sample_id = item['id']
                    complex_prompt = item['complex']
                    simple_prompt = item['simple']

                    complex_image, simple_image = generate_image_pair(
                        model, 
                        seed=args.seed, 
                        prompts=[complex_prompt, simple_prompt])
                    
                    complex_image.save(f"{complex_imgs_dir}/{sample_id}.png")
                    simple_image.save(f"{simple_imgs_dir}/{sample_id}.png")
                    print(f"第 {sample_id} 个 prompt的图像保存完成!")

                    # print("Complex Image Shape:", complex_image.shape)
                    pred_act1, pred_act2, start_idx = benchmark_metric.get_pred_feats(complex_image, simple_image, pred_act1, pred_act2, start_idx)

                # c. compute_statistics
                # shape: [10,2048]
                complex_statistics = benchmark_metric.compute_statistics(pred_act1)
                simple_statistics = benchmark_metric.compute_statistics(pred_act2)
                fid_value = complex_statistics.frechet_distance(simple_statistics)
                
                print("FID:", fid_value)



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
        default="FID",
        type=str,
        help="ImageReward-v1.0, Aesthetic, BLIP or CLIP, splitted with comma(,) if there are multiple benchmarks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--gpu_id",
        default=None,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )

    args = parser.parse_args()

    test_benchmark(args)