import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterGuidedChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(CenterGuidedChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        self.shared_MLP = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def get_scales(self, size):
        scales = []
        for i in range(1, size + 1, 2):
            scales.append(i)
        return scales

    def multi_scale_pooling(self, x, scales):
        B, C, H, W = x.size()
        pooled_features = []
        for scale in scales:
            if scale == min(H, W):
                pooled = F.adaptive_avg_pool2d(x, (1, 1))
            else:
                center_h, center_w = H // 2, W // 2
                half_scale = scale // 2
                start_h = max(0, center_h - half_scale)
                end_h = min(H, center_h + half_scale + 1)
                start_w = max(0, center_w - half_scale)
                end_w = min(W, center_w + half_scale + 1)
                center_region = x[:, :, start_h:end_h, start_w:end_w]
                pooled = F.adaptive_avg_pool2d(center_region, (1, 1))
            pooled = pooled.squeeze(-1).squeeze(-1)
            pooled_features.append(pooled)
        stacked_features = torch.stack(pooled_features, dim=1)
        return stacked_features

    def forward(self, x):
        B, C, H, W = x.size()
        max_size = min(H, W)
        scales = self.get_scales(max_size)
        multi_scale_features = self.multi_scale_pooling(x, scales)
        mlp_output = self.shared_MLP(multi_scale_features)
        attention_weights = torch.mean(mlp_output, dim=1)
        attention_weights = self.sigmoid(attention_weights)
        attention_map = attention_weights.unsqueeze(-1).unsqueeze(-1)
        return attention_map * x


class CenterGuidedSpatialAttention(nn.Module):
    def __init__(self, n, kernel_size=1, k=10):
        super(CenterGuidedSpatialAttention, self).__init__()
        self.k = k
        self.conv = nn.Conv2d(k, 1, kernel_size=kernel_size)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        center_features = x[:, :, height // 2, width // 2]
        _, topk_indices = torch.topk(center_features, self.k, dim=1)
        topk_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        topk_feature_maps = torch.gather(x, 1, topk_indices)
        attention_logits = self.conv(topk_feature_maps)
        attention_map = torch.sigmoid(attention_logits)
        attended_features = x * attention_map
        return attended_features


class CGCBAMBlock(nn.Module):
    def __init__(self, num_channels, n, reduction_ratio=16, kernel_size=3):
        super(CGCBAMBlock, self).__init__()
        self.ca = CenterGuidedChannelAttention(num_channels, reduction_ratio)
        self.sa = CenterGuidedSpatialAttention(n, kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n, reduction_ratio=16, kernel_size=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.cbam = CGCBAMBlock(out_channels, n, reduction_ratio, kernel_size)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        x = x + residual
        return x


class CGCBACNN(nn.Module):
    def __init__(self, inputdim, n, num_classes=10, num_residual_blocks=2, dim=128):
        super(CGCBACNN, self).__init__()
        self.conv1 = ConvBlock(inputdim, dim)
        self.blocks = nn.ModuleList([ResidualBlock(dim, dim, n) for _ in range(num_residual_blocks)])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
