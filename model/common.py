import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.autograd import Variable, Function

def default_conv(in_channels, out_channels, kernel_size, dilation=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        dilation=dilation,
        padding=(kernel_size - 1) // 2, bias=bias)

class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)

    
class Downsampler(nn.Sequential):
     def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, n_feats//4, 3, bias))
                m.append(Shuffle_d(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, n_feats//9, 3, bias))
            m.append(Shuffle_d(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*m)



class Upsampler(nn.Sequential):
    
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class pixelConv(nn.Module):
    # Generate pixel kernel  (3*k*k)xHxW
    def __init__(self,in_feats,out_feats=3, rate=4, ksize=3):
        super(pixelConv,self).__init__()
        self.padding = (ksize-1)//2
        self.ksize = ksize
        self.zero_padding = nn.ZeroPad2d(self.padding)
        mid_feats = in_feats*rate**2
        self.kernel_conv =nn.Sequential(*[
            nn.Conv2d(in_feats,mid_feats,kernel_size=3,padding=1),
            nn.Conv2d(mid_feats,mid_feats,kernel_size=3,padding=1),
            nn.Conv2d(mid_feats,3*ksize**2,kernel_size=3,padding=1)
        ])
  
    def forward(self, x_feature, x):
        
        kernel_set = self.kernel_conv(x_feature)

        dtype = kernel_set.data.type() 
        ks = self.ksize
        N = self.ksize**2 # patch size 
        # padding the input image with zero values
        if self.padding:
            x = self.zero_padding(x)
            
        p = self._get_index(kernel_set,dtype)
        p = p.contiguous().permute(0, 2, 3, 1).long()

        # p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)],
        #                  dim=-1).long()
       
        #(b,3,h,w,ksize**2)
        x_pixel_set = self._get_x_q(x, p, N)

        
        b,c,h,w = kernel_set.size()
        kernel_set_reshape = kernel_set.reshape(-1,self.ksize**2,3,h,w).permute(0,2,3,4,1)
        x_ = x_pixel_set
     
        out = x_*kernel_set_reshape
        out = out.sum(dim=-1,keepdim=True).squeeze(dim=-1)
        out = out
        return out 

    def _get_index(self, kernel_set, dtype):
        '''
        get absolute index of each pixel in image
        '''
        N, b, h, w = self.ksize**2, kernel_set.size(0), kernel_set.size(2), kernel_set.size(3) # 坐标包含 ksize*ksize 个(x,y)
        # get absolute index of center index
        p_0_x, p_0_y = np.meshgrid(range(self.padding, h + self.padding), range(self.padding, w + self.padding), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)  # 以并联的形式 前 N 个 x轴 坐标、后 N 个 y轴 坐标
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        # get relative index around center pixel
        p_n_x, p_n_y = np.meshgrid(range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1),
                                   range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))  # 以并联的形式 前 N 个 x轴 坐标、后 N 个 y轴 坐标
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False) 
        p = p_0 + p_n
        p = p.repeat(b,1,1,1)
        return p
    def _get_x_q(self, x, q, N):
        '''
        返回给定坐标矩阵中对应的特征图中的特征值
        '''
        b, h, w, _ = q.size()  # dimension of q: (b,h,w,2N)
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*padded_w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        # index_x*w + index_y 变换到1维之后的真实索引 其中索引大小小于 index_x<= w-1 index_y <= H-1
        index = q[..., :N] * padded_w + q[...,N:] 

        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

class MeanBlur(nn.Module):
    def __init__(self):
        super(MeanBlur, self).__init__()
        kernel = np.array([[1./25.,1./25., 1./25.,1./25., 1./25.],
                      [1./25.,1./25., 1./25.,1./25., 1./25.],
                       [1./25.,1./25., 1./25.,1./25., 1./25.],
                          [1./25.,1./25., 1./25.,1./25., 1./25.],
                          [1./25.,1./25., 1./25.,1./25., 1./25.]])
       
        kernel = torch.FloatTensor(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)
        self.mean = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2,groups=3,bias=False)
        self.mean.weight = nn.Parameter(kernel, requires_grad=False)
   
    def forward(self, x):
        x = self.mean(x)
        return x
    

class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = np.array([[1./256., 4./256., 6./256., 4./256., 1./256.],
                        [4./256., 16./256., 24./256., 16./256., 4./256.],
                        [6./256., 24./256., 36./256., 24./256., 6./256.],
                        [4./256., 16./256., 24./256., 16./256., 4./256.],
                        [1./256., 4./256., 6./256., 4./256., 1./256.]])
       
        kernel = torch.FloatTensor(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)
        self.gaussian = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2,groups=3,bias=False)
        self.gaussian.weight = nn.Parameter(kernel, requires_grad=False)
 
    def forward(self, x):
        x = self.gaussian(x)
        return x

class Laplacian_pyramid(nn.Module):
    def __init__(self,step=3):
        super(Laplacian_pyramid, self).__init__()
        self.Gau = GaussianBlur()
        self.mean = MeanBlur()
        self.step = step
        self.pool = nn.AvgPool2d(2,stride=2)
    def forward(self, x):
        Gaussian_lists = [x]
        Laplacian_lists= []
        for _ in range(self.step-1):
            gaussian_down = self.Prdown(Gaussian_lists[-1])
            Gaussian_lists.append(gaussian_down)
            Lap = Gaussian_lists[-2]-self.PrUp(Gaussian_lists[-1])
            Laplacian_lists.append(Lap)
        return Gaussian_lists,Laplacian_lists

    def Prdown(self,x):
        x_ = self.Gau(x)
        x_ = self.pool(x_)
#         x_ = x_[:,:,::2,::2]
        return x_

    def PrUp(self,x):
        b,c,h,w = x.size()
#         up_x = torch.zeros((b,c,h*2,w*2),device='cuda')
#         up_x[:,:,::2,::2]= x
        up_x = F.interpolate(x,scale_factor=2,mode='bilinear')
        up_x = self.Gau(up_x)
        
        return up_x

class Laplacian_reconstruction(nn.Module):
    def __init__(self):
        super(Laplacian_reconstruction, self).__init__()
        self.Gau = GaussianBlur()
        self.mean = MeanBlur()
    def forward(self, x_lap,x_gau):
        b,c,h,w = x_gau.size()
#         up_x = torch.zeros((b,c,h*2,w*2),device='cuda')
#         up_x[:,:,::2,::2]= x_gau
        up_x = F.interpolate(x_gau,scale_factor=2,mode='bilinear')
        up_x = self.Gau(up_x) + x_lap
        return up_x
    

if __name__ == "__main__":
    pixel_conv = pixelConv(in_feats=16, out_feats=3, ksize=5, rate=1) 
    x =  torch.ones((4,3,10,10))
    x1 = torch.ones((4,16,10,10))
    x = pixel_conv(x1, x)
