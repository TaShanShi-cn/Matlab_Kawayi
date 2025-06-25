# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random

import numpy as np

import torch
import torchvision.utils as tvu #图像工具，常用来保存或拼接图片

from unet_ddpm import Model #导入U-net模型类，用于DDPM中噪声预测


#生成一个线性变化的beta序列
def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


#根据时间步t从a中抽取对应系数，并reshape成与x_shape广播兼容的形状
def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape #batch size
    assert x_shape[0] == bs #确保batch size匹配
    # 从a中按t的索引提取对应元素，a转换为tensor且放到与t相同的设备上
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,) #确认输出形状正确
    # reshape成(bs,1,1,1,...)以广播给x
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


# 根据当前x和时间t，结合模型输出，计算从x_t采样得到x_{t-1}的噪声去除步骤
def image_editing_denoising_step_flexible_mask(x, t, *, model, logvar, betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0) #累计乘积，表示alpha的累计量

    model_output = model(x, t) #输入当前图像x和时间t，模型预测噪声，是预测的噪声
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape) #根据t提取log方差
    noise = torch.randn_like(x) #生成与x形状相同的标准正态噪声
    mask = 1 - (t == 0).float() #t=0时mask=0，表示最后一步不加噪声
    #reshape成与x兼容形状
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


# DDPM扩散过程类
class Diffusion(torch.nn.Module):
    def __init__(self, args, config, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        print("Loading model")
        if self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        else:
            raise ValueError

        model = Model(self.config) #初始化UNet
        ckpt = torch.hub.load_state_dict_from_url(url, map_location='cpu') #下载权重
        model.load_state_dict(ckpt)
        model.eval()

        self.model = model

        self.model_var_type = config.model.var_type #模型方差类型
        # 获取beta序列（扩散时间步长）
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

        #计算alpha序列及其累计乘积
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        #后验方差计算公式
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        #根据方差类型设置log方差
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    #采样函数，输入初始图像，返回扩散过程采样结果
    def image_editing_sample(self, img=None, bs_id=0, tag=None):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        with torch.no_grad():
            # 生成tag目录
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim # 确认图片维度为4维（batch,C,H,W）
            x0 = img

            # bs_id小于2时，创建目录并保存原图
            if bs_id < 2:
                os.makedirs(out_dir, exist_ok=True)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))

            xs = []
            # 按采样步数循环
            for it in range(self.args.sample_step):
                e = torch.randn_like(x0) #生成随机噪声e
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0).to(x0.device) #alpha累计乘积
                # 根据DDPM前向扩散公式，产生带噪图像x
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                #保存初始带噪图像
                if bs_id < 2:
                    tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

                # 反向扩散去噪过程，倒序采样
                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=img.device)
                    x = image_editing_denoising_step_flexible_mask(x, t=t, model=self.model,
                                                                    logvar=self.logvar,
                                                                    betas=self.betas.to(img.device))
                    # added intermediate step vis
                    # 保存中间采样结果，每隔50步保存一次
                    if (i - 49) % 50 == 0 and bs_id < 2:
                        tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'noise_t_{i}_{it}.png'))

                x0 = x

                #保存最终采样结果
                if bs_id < 2:
                    torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

                xs.append(x0)

            #返回拼接的所有采样结果tensor
            return torch.cat(xs, dim=0)
