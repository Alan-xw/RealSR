import torch
from torch import nn
import numpy as np
from torch.autograd import Variable, Function
from torch.nn import functional as F
from model.common import *


def make_model(args, parent=False):
    return RealSR(args)


class basicblock(nn.Module):
    def __init__(self,in_feats,out_feats,ksize=3):
        super(basicblock,self).__init__()
        block = [
            nn.Conv2d(in_feats, out_feats, kernel_size=ksize, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=ksize, padding=1),        
        ]
        self.block = nn.Sequential(*block)
    def forward(self, x):
        return self.block(x)


class backbone(nn.Sequential):
    def __init__(self,in_feats,out_feats,n_blocks=8, ksize=3):
        m = [nn.Conv2d(in_feats,out_feats,kernel_size=ksize, padding=1)]
        for _ in range(n_blocks):
            m.append(basicblock(out_feats,out_feats,ksize))
        m.append(nn.Conv2d(out_feats,out_feats,kernel_size=ksize, padding=1))
        super(backbone, self).__init__(*m)


class RealSR(nn.Module):
    def __init__(self, args):
        super(RealSR,self).__init__()
        n_colors = args.n_colors
        n_feats = args.n_feats

        self.up = nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=None)
        self.shuffle_up_4= nn.PixelShuffle(4)
        self.shuffle_up_2= nn.PixelShuffle(2)
        self.head = nn.Sequential(*[
            nn.Conv2d(n_colors,n_feats//4,3,padding=1,stride=1),
            nn.Conv2d(n_feats//4,n_feats//16,3,padding=1,stride=1),
            Shuffle_d(scale = 4)
        ])
        self.backbone = backbone(n_feats, n_feats)
        self.laplacian = Laplacian_pyramid()
        self.laplacian_rec = Laplacian_reconstruction()
        # Branches
        self.branch_1 = pixelConv(in_feats=n_feats//16, out_feats=3, rate=4, ksize=5) 
        self.branch_2 = pixelConv(in_feats=n_feats//4, out_feats=3, rate=2, ksize=5) 
        self.branch_3 = pixelConv(in_feats=n_feats, out_feats=3, rate=1, ksize=5) 

    def forward(self, x):
          
        # Lapcian Pyramid

        Gaussian_lists,Laplacian_lists = self.laplacian(x)
        Lap_1,Lap_2 = Laplacian_lists[0],Laplacian_lists[1]
        Gaussian_3 = Gaussian_lists[-1]
        
        f_ = self.head(x)
        f_ = self.backbone(f_)  
        # branch_1
        f_1 = self.shuffle_up_4(f_)
        out_1 = self.branch_1(f_1,Lap_1)
        out_1 = 1.2 * out_1 
        # branch_2
        f_2 = self.shuffle_up_2(f_)
        out_2 = self.branch_2(f_2, Lap_2)
        # branch_3
        out_3 = self.branch_3(f_, Gaussian_3)        
        
        # Laplacian Reconstruction
        merge_x2 = self.laplacian_rec(out_2,out_3)
        merge = self.laplacian_rec(out_1,merge_x2)
        
        return merge

if __name__ == "__main__":
    # pixel_conv = RealSR() 
  
    x =  torch.rand((4,3,4,4))
    x_1= x.mean(dim=-1,keepdim=True).mean(dim=-2,keepdim=True)
    x_2 = x.mean(dim=(2,3),keepdim=True)
    print(x_1[0,0],x_2[0,0])
    # x = pixel_conv(x)
