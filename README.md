# CrystalDiffusion

## Environment

You can run ```!pip install -r requirement.txt``` to install packages required.


## Usage

[npz2cloud.py](PCCD/preprocess/npz2cloud.py)and [npz2cloud.py](PCCD/preprocess/npz2cloud.py) in PCCD/preprocess is for data preprocessing.

run ```python train.py``` to train PCCD.

If you want use PCCD, you can run:

    python train.py

You can also use it as follows:
```python
from PCCD.Unet import Unet
from PCCD.DDPM import GaussianDiffusion
import torch
from PCCD.process import *
import numpy as np


sh = 1
image_classes = torch.Tensor([5]).to(torch.long).cuda()

iss0 = torch.Tensor().to(torch.long).cuda()
elem = torch.Tensor([[m['Ca'],m['Mg'],m['O']]]).to(torch.long).cuda()
el = ['Ca','Mg','O']

ee = torch.Tensor([1]).to(torch.long).cuda()


sampled_images = diffusion.sample(
    classes=image_classes,
    e=ee,
    iss=iss0,
    cond_scale=3.
)

data = sampled_images.to('cpu').detach().numpy()
process('./', sh, data, el)
```
## Citation

    @misc{denoising-diffusion-pytorch ,
        author  = {lucidrains},
        url     = {https://github.com/lucidrains/denoising-diffusion-pytorch}
    }

    @article{LI2025111659,
    title = {Generative design of crystal structures by point cloud representations and diffusion model},
    journal = {iScience},
    volume = {28},
    number = {1},
    pages = {111659},
    year = {2025},
    issn = {2589-0042},
    doi = {https://doi.org/10.1016/j.isci.2024.111659},
    url = {https://www.sciencedirect.com/science/article/pii/S2589004224028864},
    author = {Zhelin Li and Rami Mrad and Runxian Jiao and Guan Huang and Jun Shan and Shibing Chu and Yuanping Chen},
    }
