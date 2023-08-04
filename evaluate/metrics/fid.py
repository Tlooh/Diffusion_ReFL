import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import warnings
from scipy import linalg

from inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePairDataset(Dataset):
    def __init__(self, complex_path, simple_path, transform=None):
        self.complex_files = sorted([os.path.join(complex_path, file) for file in os.listdir(complex_path) if file.endswith(tuple(IMAGE_EXTENSIONS))])
        self.simple_files = sorted([os.path.join(simple_path, file) for file in os.listdir(simple_path) if file.endswith(tuple(IMAGE_EXTENSIONS))])
        self.transform = transform

    def __len__(self):
        return min(len(self.complex_files), len(self.simple_files))

    def __getitem__(self, idx):
        # print("Complex Image Path:", self.complex_files[idx])
        # print("Simple Image Path:", self.simple_files[idx])
        complex_img = Image.open(self.complex_files[idx]).convert('RGB')
        simple_img = Image.open(self.simple_files[idx]).convert('RGB')

        if self.transform is not None:
            complex_img = self.transform(complex_img)
            simple_img = self.transform(simple_img)

        return complex_img, simple_img



class InvalidFIDException(Exception):
    pass


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(complex_img, simple_img):

    print("calculate!")


def get_pred_act(model, batch_imgs):

    with torch.no_grad():
        pred = model(batch_imgs)[0]
    
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred = pred.squeeze(3).squeeze(2).cpu().numpy() #[batch, 2048]

    return pred


def compute_statistics(activations: np.ndarray) -> FIDStatistics:
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return FIDStatistics(mu, sigma)


def main(args):

    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")

    # 1. load dataset
    complex_path = os.path.join(args.img_dir, 'complex')
    simple_path = os.path.join(args.img_dir, 'simple')

    # define transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转换为Tensor
    ])

    img_pair_dataset = ImagePairDataset(complex_path, simple_path, transform=transform)

    img_pair_dataloader = DataLoader(img_pair_dataset, batch_size=args.batch_size, shuffle=False)

    # feature store
    length = 10
    dims = 2048
    pred_act1 = np.empty((length, dims))
    pred_act2 = np.empty((length, dims))
    star_idx = 0


    # 2. calculate FID for each image_pair (complex / simple)
    # a. load inception v3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # b. get activations for each batch
    for complex_imgs, simple_imgs in tqdm(img_pair_dataloader):
        # print(complex_imgs.shape)  shape: [batch, 3, 256, 256]
        complex_imgs = complex_imgs.to(device)
        simple_imgs = simple_imgs.to(device)
        
        # get pred with passing Inception V3
        act_of_complex = get_pred_act(model, complex_imgs) 
        act_of_simple = get_pred_act(model, simple_imgs) 

        # actually, pred_of_complex.shape[0] = batch_size
        # shape: [10,2048]
        pred_act1[star_idx:star_idx + act_of_complex.shape[0]] = act_of_complex
        pred_act2[star_idx:star_idx + act_of_simple.shape[0]] = act_of_simple

        star_idx = star_idx + act_of_simple.shape[0]
    
    # c. compute_statistics
    
    complex_statistics = compute_statistics(pred_act1)
    simple_statistics = compute_statistics(pred_act2)

    print(complex_statistics)

    fid_value = complex_statistics.frechet_distance(simple_statistics)
    print("FID:", fid_value)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a script to test baselines.")

    parser.add_argument(
        "--img_dir",
        default="/home/khf/liutao/data/images",
        type=str,
        help="Path to the generated images directory. The sub-level directory name should be <complex> and <simple>",
    )
    parser.add_argument(
        "--batch_size",
        default=5,
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
    main(args)

