import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import warnings
from scipy import linalg

from inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d


# define transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
])

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
    


class FIDScore(nn.Module):
    def __init__(self, dims, device="cpu"):
        super().__init__()
        self.dims = dims
        self.device = device

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception_model = InceptionV3([block_idx]).to(device)
    
    def get_activations(self, imgs):
        
        with torch.no_grad():
            pred = self.inception_model(imgs)[0]
        
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy() #[batch, 2048]

        return pred

    
    def get_pred_feats(self, complex_img, simple_img, pred_act1, pred_act2, start_idx):

        # print(complex_img.shape)  shape: [1, 3, 256, 256]
 
        complex_img = transform(complex_img).unsqueeze(0).to(self.device)
        simple_img = transform(simple_img).unsqueeze(0).to(self.device)
        self.inception_model.eval()

        # get pred with passing Inception V3
        act_of_complex = self.get_activations(complex_img)
        act_of_simple = self.get_activations(simple_img)

        # actually, pred_of_complex.shape[0] = batch_size
        pred_act1[start_idx:start_idx + act_of_complex.shape[0]] = act_of_complex
        pred_act2[start_idx:start_idx + act_of_simple.shape[0]] = act_of_simple

        start_idx = start_idx + act_of_simple.shape[0]


        return pred_act1, pred_act2, start_idx


    def compute_statistics(self, feats: np.ndarray) -> FIDStatistics:
        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)
        return FIDStatistics(mu, sigma)




# ========run_test=========
# if __name__ == "__main__":


#     # 定义图像预处理
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),  # 调整为256x256大小
#         transforms.ToTensor(),
#     ])

#     def load_images_from_dir(image_dir, data_type):
#         data_path = os.path.join(image_dir, data_type)
#         image_list = []

#         for img_name in sorted(os.listdir(data_path)):
            
#                 img_path = os.path.join(data_path, img_name)
#                 img = Image.open(img_path)
#                 # img_tensor = transform(img)
#                 image_list.append(img)

#         return image_list


#     # 指定image目录路径
#     image_dir = '/home/khf/liutao/data/images'
#     device = "cuda"

#     # 从complex和simple目录加载图像并转换为张量
#     complex_images = load_images_from_dir(image_dir, 'complex')
#     simple_images = load_images_from_dir(image_dir, 'simple')

#     num_pair_images = len(complex_images)
#     dims = 2048
#     pred_act1 = np.empty((num_pair_images, dims))
#     pred_act2 = np.empty((num_pair_images, dims))
#     start_idx = 0
#     # 打印图像tensor形式
#     benchmark_metric = FIDScore(device=device, dims=2048).to(device)

#     for complex_image, simple_image in zip(complex_images, simple_images):
#         # print(pred_act1)
#         pred_act1, pred_act2, start_idx = benchmark_metric.get_pred_feats(complex_image, simple_image, pred_act1, pred_act2, start_idx)
        
#         # print("Complex Image Shape:", complex_image.shape)
#         # print("pred_act1 Shape:", pred_act1.shape)
    
    
#     complex_statistics = benchmark_metric.compute_statistics(pred_act1)
#     simple_statistics = benchmark_metric.compute_statistics(pred_act2)
#     # print(pred_act1.shape)
#     print(complex_statistics)
#     fid_value = complex_statistics.frechet_distance(simple_statistics)
    
#     print("FID:", fid_value)




