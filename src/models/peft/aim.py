import torch.nn as nn

class AIMAdapter(nn.Module):
    def __init__(self, D_features, bottleneck_dim, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, bottleneck_dim)
        self.D_fc2 = nn.Linear(bottleneck_dim, D_features)
        nn.init.zeros_(self.D_fc2.weight)
        nn.init.zeros_(self.D_fc2.bias)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        return x + xs if self.skip_connect else xs
