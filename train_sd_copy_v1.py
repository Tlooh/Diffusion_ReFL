import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers


from PIL import Image
from  MyReward import MyReward as RM

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "prompts": ("complex", "simple"),
}



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="/home/khf/liutao/data/prompts/prompts.json",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--complex_caption_column", type=str, default="complex", help="The column of the dataset containing a caption or a list of complex caption."
    )
    parser.add_argument(
        "--simple_caption_column",
        type=str,
        default="simple",
        help="The column of the dataset containing a caption or a list of simple captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_path is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    med_config = "/home/khf/liutao/Diffusion_ReFL/configs/bert_config.json"
    reward_model = RM(med_config=med_config, device=accelerator.device).to(accelerator.device)
    reward_model.eval()

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    reward_model.requires_grad_(False)


    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        data_files["train"] = args.train_data_path
        dataset = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names
    # print(column_names)
    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.complex_caption_column is None:
        complex_caption_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        complex_caption_column = args.complex_caption_column
        if complex_caption_column not in column_names:
            raise ValueError(
                f"--complex_caption_column' value '{args.complex_caption_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.simple_caption_column is None:
        simple_caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        simple_caption_column = args.simple_caption_column
        if simple_caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.simple_caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, caption_name = "complex",is_train=True):
        captions = []
        if caption_name == "complex":
            caption_column = "complex"
        else:
            caption_column = "simple"

        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def preprocess_train(examples):
       
        examples["complex_input_ids"] = tokenize_captions(examples, caption_name="complex")
        
        examples["simple_input_ids"] = tokenize_captions(examples,caption_name="simple")

        return examples


    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        # train_dataset = dataset["train"]
        train_dataset = dataset["train"].with_transform(preprocess_train)
    
    # # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device,)
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    for example in train_dataset:
        id = example["id"]
        complex_prompt = example["complex"]
        simple_prompt = example["simple"]
        # complex_input_ids = example["complex_input_ids"]
        # simple_input_ids = example["simple_input_ids"]
        
        print(f"ID: {id}")
        print(f"Complex prompt: {complex_prompt}")
        print(f"Simple prompt: {simple_prompt}")
        # print(f"Complex input_ids: {complex_input_ids}")
        # print(f"Simple input_ids: {simple_input_ids}")
        print("=" * 30)

        # complex_input_ids = tokenizer(complex_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids

        # simple_input_ids = tokenizer(simple_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids

        # print(complex_input_ids)
        # print(complex_input_ids.shape)

        complex_input_ids = example["complex_input_ids"].unsqueeze(0)
        simple_input_ids = example["simple_input_ids"].unsqueeze(0)
        print(complex_input_ids.shape)

        complex_encoder_hidden_states = text_encoder(complex_input_ids.to(accelerator.device))[0]
        simple_encoder_hidden_states = text_encoder(simple_input_ids.to(accelerator.device))[0]
        cond_prompt_embeds = torch.cat((complex_encoder_hidden_states, simple_encoder_hidden_states),dim=0)

        uncond_input = tokenizer(
        [""] * 2, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(accelerator.device),
                uncond_input.attention_mask.to(accelerator.device)
                )[0]
        
        text_embeddings = torch.cat([negative_prompt_embeds, cond_prompt_embeds])
        print(text_embeddings.shape)

        # we generate the initial random noise.
        guidance_scale = 7.5
        latents = torch.randn(
            (2, 4, 64, 64)
        )
        latents = latents.to(accelerator.device)
        noise_scheduler.set_timesteps(50, device=accelerator.device)
        timesteps = noise_scheduler.timesteps

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        

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
        print("save successfully!")


    # def collate_fn(examples):
    #     complex_input_ids = torch.stack([example["complex_input_ids"] for example in examples])
    #     simple_input_ids = torch.stack([example["simple_input_ids"] for example in examples])

    #     complex_input_ids = complex_input_ids.view(-1, complex_input_ids.shape[-1])
    #     simple_input_ids = simple_input_ids.view(-1, simple_input_ids.shape[-1])

    #     return {"complex_input_ids": complex_input_ids, "simple_input_ids": simple_input_ids}
    
    # train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         shuffle=True,
    #         collate_fn=collate_fn,
    #         batch_size=args.train_batch_size,
    #         num_workers=args.dataloader_num_workers,
    #     )
    
    
    # # Scheduler and math around the number of training steps.
    # overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )

    # # Prepare everything with our `accelerator`.
    # unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     unet, optimizer, train_dataloader, lr_scheduler
    # )

    # if args.use_ema:
    #     ema_unet.to(accelerator.device)

    # # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # # as these weights are only used for inference, keeping weights in full precision is not required.
    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    #     args.mixed_precision = accelerator.mixed_precision
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    #     args.mixed_precision = accelerator.mixed_precision

    # # Move text_encode and vae to gpu and cast to weight_dtype
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    # vae.to(accelerator.device, dtype=weight_dtype)
    # reward_model.to(accelerator.device, dtype=weight_dtype)

    # # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     tracker_config = dict(vars(args))
    #     tracker_config.pop("validation_prompts")
    #     accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # # Train!
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    # logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # global_step = 0
    # first_epoch = 0

    # # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(args.output_dir)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         accelerator.print(
    #             f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
    #         )
    #         args.resume_from_checkpoint = None
    #     else:
    #         accelerator.print(f"Resuming from checkpoint {path}")
    #         accelerator.load_state(os.path.join(args.output_dir, path))
    #         global_step = int(path.split("-")[1])

    #         resume_global_step = global_step * args.gradient_accumulation_steps
    #         first_epoch = global_step // num_update_steps_per_epoch
    #         resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    # # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar.set_description("Steps")

    # guidance_scale = 7.5
    # do_classifier_free_guidance = guidance_scale > 1.0

    # for epoch in range(first_epoch, args.num_train_epochs):
    #     unet.train()
    #     train_loss = 0.0
    #     for step, batch in enumerate(train_dataloader):
    #         # Skip steps until we reach the resumed step
    #         if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
    #             if step % args.gradient_accumulation_steps == 0:
    #                 progress_bar.update(1)
    #             continue
            
    #         print("step: ", step,"\n","batch:",batch)
    #         # print("step: ", step)

          
    #         complex_encoder_hidden_states = text_encoder(batch["complex_input_ids"])[0]
    #         print(complex_encoder_hidden_states.shape)
    #         simple_encoder_hidden_states = text_encoder(batch["simple_input_ids"])[0]
            
    #         cond_prompt_embeds = torch.cat((complex_encoder_hidden_states, simple_encoder_hidden_states), dim=0)

    #         print(cond_prompt_embeds.shape)

    #         uncond_input = tokenizer(
    #         [""] * 2, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    #         negative_prompt_embeds = text_encoder(
    #             uncond_input.input_ids.to(accelerator.device),
    #             uncond_input.attention_mask.to(accelerator.device)
    #         )[0]

    #         prompt_embeds = torch.cat([negative_prompt_embeds, cond_prompt_embeds])
    #         print(prompt_embeds.shape)

    #         # 随机初始化 batch 个 latents
    #         # latents = torch.randn((2, 4, 64, 64), device=accelerator.device)
    #         latents = torch.randn(
    #             (1, unet.in_channels, 64, 64),
                
    #         ).to(accelerator.device)

    #         # 设置噪声模型的时间步数，从而决定模型在每个时间步长上的噪声情况。
    #         noise_scheduler.set_timesteps(50, device=accelerator.device)
    #         timesteps = noise_scheduler.timesteps

    #         for i, t in enumerate(timesteps):
    #             latent_model_input = torch.cat([latents] * 2)

    #             latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                
    #             with torch.no_grad():
                    
    #                 noise_pred = unet(
    #                     latent_model_input,
    #                     t,
    #                     encoder_hidden_states=prompt_embeds,
    #                     ).sample
                        
    #             # perform guidance
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #             # compute the previous noisy sample x_t -> x_t-1
    #             latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        

    #         pred_original_sample = latents
    #         pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
    #         image = vae.decode(pred_original_sample.to(weight_dtype)).sample
    #         image = (image / 2 + 0.5).clamp(0, 1)

    #         image = image.detach().cpu().permute(0, 2, 3, 1).numpy() # (1, 512, 512, 3)
    #         images = (image * 255).round().astype("uint8")
    #         pil_images = [Image.fromarray(image) for image in images]

    #         pil_images[0].save(f"/home/khf/liutao/data/outputs/{step}_complex.png")
    #         pil_images[1].save(f"/home/khf/liutao/data/outputs/{step}_simple.png")
    #         print("save successfully!")


if __name__ == "__main__":
    main()
