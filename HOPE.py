import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_sum
import math
from openpoints.models.layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
import numpy as np

def xyz2sphere(xyz, normalize=True):
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)
    theta = torch.acos(xyz[..., 2, None] / rho)
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])
    idx = rho == 0
    theta[idx] = 0
    if normalize:
        theta = theta / np.pi
        phi = phi / (2 * np.pi) + .5
    out = torch.cat([rho, theta, phi], dim=-1)
    return out



class HOPE(nn.Module):
    def __init__(self,
                 convs_pe,
                 in_channel,
                 out_channel,
                 group_args,
                 k,
                 attn=False,
                 global_pe=False,
                 epchannel=30,
                 efchannel=30,
                 dim=32,
                 alpha=5.0,
                 ):
        super().__init__()

        self.use_conv_pe = False
        if convs_pe != None:
            self.convs_pe = convs_pe
            self.use_conv_pe = True

        self.is_attn = attn
        self.embed_xyz = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=1),
                                       nn.BatchNorm2d(in_channel), )
        self.embed_sphere = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=1),
                                          nn.BatchNorm2d(in_channel),)
        if attn:
            self.explicit_structure = PointHop()
            self.embed_start = nn.Sequential(nn.Conv2d(in_channel * 2, dim, kernel_size=1),
                                             nn.BatchNorm2d(dim),
                                             nn.ReLU(), )
            self.k_embed = nn.Sequential(nn.Linear(efchannel, dim),nn.LayerNorm(dim),nn.ReLU(),nn.Linear(dim, dim))
            self.q_embed = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), )
            self.v_embed = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), )
            self.mlp1 = nn.Sequential(
                nn.Conv2d(dim, out_channel, kernel_size=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(), )
        else:
            self.mlp1 = nn.Sequential(
                nn.Conv2d(in_channel*2, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, out_channel, kernel_size=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
            )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_mlp = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.use_global_pe = global_pe
        if global_pe:
            self.grouper = create_grouper(group_args)
            self.embed_xyz0 = nn.Sequential(nn.Conv1d(in_channel, in_channel,kernel_size=1),
                                            nn.BatchNorm1d(in_channel), )
            self.embed_sphere0 = nn.Sequential(nn.Conv1d(in_channel, in_channel,kernel_size=1),
                                               nn.BatchNorm1d(in_channel),)
            self.pre_mlp1 = nn.Sequential(nn.Conv1d(in_channel*2, dim,kernel_size=1),
                                          nn.BatchNorm1d(dim),
                                          nn.ReLU(),
                                          nn.Conv1d(dim, out_channel,kernel_size=1),
                                          nn.BatchNorm1d(out_channel),
                                          nn.ReLU(),
                                          )
            self.alpha = torch.nn.Parameter(torch.tensor(0.95))

    def forward(self,new_p_dp_fj,p,idx):
        # ep(b,epchannel,n,k)
        # ef(b,n,efchannel)
        # dp(b,3,n,k)
        # new_p (b,n,3)
        # p(b,n0,3)
        new_p, dp, fj = new_p_dp_fj

        dp_xyz = self.embed_xyz(dp)
        dp_sphere = self.embed_sphere(xyz2sphere(dp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = torch.cat([dp_xyz, dp_sphere], dim=1)

        if self.is_attn:
            x = self.embed_start(x)
            ef = self.explicit_structure(dp, new_p)  # (b,n,h)
            k = self.k_embed(ef)  # (b,n,h)
            q = self.q_embed(x)  # (b,n,h,k)
            v = self.v_embed(x)  # (b,h,n,k)
            qkrel = torch.einsum('bnhk,bnha->bnka', q.permute(0, 2, 1, 3), k.unsqueeze(-1)).squeeze(-1)  # (b,n,k)
            a1 = F.softmax(qkrel, dim=-1).unsqueeze(1)  # (b,n,k)->(b,1,n,k)
            x = x + v * a1  # (b,h,n,k)
            eape = self.mlp1(x)  # (b,o,n,k)
            a2 = F.sigmoid(self.channel_mlp(self.gap(eape).squeeze(-1).transpose(1, 2)).transpose(1, 2)).unsqueeze(-1)  # (b,o,1)->(b,o,1,1)
            eape = eape * a2
        else:
            eape = self.mlp1(x)

        if self.use_global_pe:
            p_xyz = self.embed_xyz0(p.permute(0,2,1))  # (b,3,n0)
            p_sphere = self.embed_sphere0(xyz2sphere(p).permute(0,2,1))  # (b,3,n0)
            p_xyz_sphere = torch.cat([p_xyz, p_sphere], dim=1)  # (b,6,n0)
            ape = self.pre_mlp1(p_xyz_sphere)  # (b,o,n0)
            new_ape = torch.gather(ape, -1, idx.unsqueeze(1).expand(-1, ape.shape[1], -1))  # (b,o,n)
            _, group_ape = self.grouper(new_p, p, ape)  # (b,o,n,k)
            eape0 = group_ape - new_ape.unsqueeze(-1)  # (b,o,n,k)
            eape = self.alpha*eape + (1-self.alpha)*eape0

        if fj == None:
            return eape
        else:
            if self.use_conv_pe:
                pe = self.convs_pe(dp)
                f = fj * pe + eape
            else:
                f = fj + eape

            return f, eape

#explicit structure
class PointHop(nn.Module):
    def __init__(self, args=None) -> None:
        super().__init__()
        self.outchannel = 6 + 24

    def forward(self, group_xyz, new_xyz, group_idx=None):
        group_xyz = group_xyz.permute(0, 2, 3, 1)
        B, N, K, C = group_xyz.shape
        X = group_xyz
        X = X.reshape(B * N, K, C)
        std_xyz = torch.std(group_xyz, dim=2, keepdim=True).view(B * N, 3)
        center = new_xyz.view(B * N, 3)
        idx = (X[:, :, 0] > 0).float() * 4 + (X[:, :, 1] > 0).float() * 2 + (X[:, :, 2] > 0).float()
        current_features = torch.zeros(B * N, 8, 3).to(group_xyz.device)
        current_features = scatter_mean(X, idx.long(), dim=1, out=current_features).view(B * N, 24)
        features = torch.cat([std_xyz, center, current_features], dim=-1)
        features = features.view(B, N, -1)
        return features  # (b,n,30)


