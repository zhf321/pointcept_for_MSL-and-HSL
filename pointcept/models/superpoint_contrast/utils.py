import torch.nn as nn
from timm.models.layers import trunc_normal_


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=256, bottleneck_dim=128):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    

class Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, hidden_dim=256, bottleneck_dim=128, mode="linear_relu"):
        """
        mode: "linear_relu" (linear + ReLU),
              "linear_bn_relu" (linear + BatchNorm + ReLU),
              "linear_bn_relu_linear" (linear + BatchNorm + ReLU + linear)
              "linear"
        """
        super().__init__()
        
        # 根据 mode 创建不同的网络层
        if mode == "linear_relu":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU() # nn.ReLU()
            )
        elif mode == "linear_bn_relu":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(),
            )
        elif mode == "linear":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.BatchNorm1d(out_dim),
                # nn.ReLU(),
            )
        elif mode == "linear_bn_relu_linear":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        elif mode == "linear_bn_relu_linear_bn_relu_linear":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.LeakyReLU(),
                nn.Linear(bottleneck_dim, out_dim)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 初始化权重
        self.apply(self._init_weights)
        


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.mlp(x)
        return x
