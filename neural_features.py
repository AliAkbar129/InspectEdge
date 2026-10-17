"""Attributed frozen ImageNet ResNet18 patch features; not full PatchCore."""
from functools import lru_cache
import numpy as np
import torch
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
@lru_cache(maxsize=1)
def encoder():
    m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    for p in m.parameters():p.requires_grad_(False)
    return m

def patches(image,size=128):
    a=np.asarray(image.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
    x=torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0)
    x=(x-torch.tensor([.485,.456,.406]).view(1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,3,1,1)
    m=encoder()
    with torch.inference_mode():
        x=m.maxpool(m.relu(m.bn1(m.conv1(x))));x=m.layer1(x);a=m.layer2(x);b=m.layer3(a)
        a=torch.nn.functional.avg_pool2d(a,3,stride=1,padding=1);b=torch.nn.functional.avg_pool2d(b,3,stride=1,padding=1)
        a=torch.nn.functional.adaptive_avg_pool2d(a,b.shape[-2:]);z=torch.cat([a,b],dim=1)
    return z[0].permute(1,2,0).reshape(-1,z.shape[1]).numpy()
