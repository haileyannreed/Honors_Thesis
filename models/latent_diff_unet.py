"""
Latent Difference U-Net with Configurable Skip Connections
Novel architecture: Separate encoders → Latent difference → Shared decoder

Supports 4 skip connection modes for ablation study:
- 'high': Use high-fidelity encoder skips (most intuitive)
- 'low': Use low-fidelity encoder skips
- 'both': Concatenate both high and low skips (most expressive)
- 'avg': Average high and low skips
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class LatentDiffUNet(nn.Module):
    """
    Dual-encoder architecture with latent space difference fusion.

    Args:
        in_channels: Input image channels (default 3)
        out_channels: Output segmentation classes (default 2)
        init_features: Initial feature channels (default 32)
        skip_mode: Skip connection mode - 'high', 'low', 'both', or 'avg' (default 'high')
    """

    def __init__(self, in_channels=3, out_channels=2, init_features=32, skip_mode='high'):
        super(LatentDiffUNet, self).__init__()

        assert skip_mode in ['high', 'low', 'both', 'avg'], \
            "skip_mode must be one of: 'high', 'low', 'both', 'avg'"

        self.skip_mode = skip_mode
        features = init_features

        # HIGH-FIDELITY ENCODER
        self.encoder1_high = self._make_block(in_channels, features, "high_enc1")
        self.pool1_high = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_high = self._make_block(features, features * 2, "high_enc2")
        self.pool2_high = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_high = self._make_block(features * 2, features * 4, "high_enc3")
        self.pool3_high = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_high = self._make_block(features * 4, features * 8, "high_enc4")
        self.pool4_high = nn.MaxPool2d(kernel_size=2, stride=2)

        # LOW-FIDELITY ENCODER
        self.encoder1_low = self._make_block(in_channels, features, "low_enc1")
        self.pool1_low = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_low = self._make_block(features, features * 2, "low_enc2")
        self.pool2_low = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_low = self._make_block(features * 2, features * 4, "low_enc3")
        self.pool3_low = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_low = self._make_block(features * 4, features * 8, "low_enc4")
        self.pool4_low = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK processes the latent difference
        self.bottleneck = self._make_block(features * 8, features * 16, "bottleneck")

        # SHARED DECODER
        # Decoder input channels depend on skip_mode
        skip_multiplier = 3 if skip_mode == 'both' else 2  # both=3x, others=2x

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._make_block(features * 8 * skip_multiplier, features * 8, "dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._make_block(features * 4 * skip_multiplier, features * 4, "dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._make_block(features * 2 * skip_multiplier, features * 2, "dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._make_block(features * skip_multiplier, features, "dec1")

        # Final segmentation head
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)

    def _get_skip_features(self, enc_high, enc_low):
        """
        Combine skip connections based on skip_mode.

        Args:
            enc_high: Features from high-fidelity encoder
            enc_low: Features from low-fidelity encoder

        Returns:
            Combined skip features
        """
        if self.skip_mode == 'high':
            return enc_high
        elif self.skip_mode == 'low':
            return enc_low
        elif self.skip_mode == 'both':
            return torch.cat([enc_high, enc_low], dim=1)
        elif self.skip_mode == 'avg':
            return (enc_high + enc_low) / 2.0
        else:
            raise ValueError(f"Unknown skip_mode: {self.skip_mode}")

    def forward(self, img_high, img_low):
        """
        Forward pass with latent space difference.

        Args:
            img_high: High-fidelity image [B, 3, H, W]
            img_low: Low-fidelity image [B, 3, H, W]

        Returns:
            Segmentation logits [B, out_channels, H, W]
        """
        # ENCODE HIGH-FIDELITY
        enc1_high = self.encoder1_high(img_high)
        enc2_high = self.encoder2_high(self.pool1_high(enc1_high))
        enc3_high = self.encoder3_high(self.pool2_high(enc2_high))
        enc4_high = self.encoder4_high(self.pool3_high(enc3_high))
        pooled_high = self.pool4_high(enc4_high)

        # ENCODE LOW-FIDELITY
        enc1_low = self.encoder1_low(img_low)
        enc2_low = self.encoder2_low(self.pool1_low(enc1_low))
        enc3_low = self.encoder3_low(self.pool2_low(enc2_low))
        enc4_low = self.encoder4_low(self.pool3_low(enc3_low))
        pooled_low = self.pool4_low(enc4_low)

        # COMPUTE DIFFERENCE IN LATENT SPACE
        latent_diff = pooled_high - pooled_low

        # BOTTLENECK processes the difference
        bottleneck = self.bottleneck(latent_diff)

        # DECODE with configurable skip connections
        dec4 = self.upconv4(bottleneck)
        skip4 = self._get_skip_features(enc4_high, enc4_low)
        dec4 = torch.cat([dec4, skip4], dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        skip3 = self._get_skip_features(enc3_high, enc3_low)
        dec3 = torch.cat([dec3, skip3], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        skip2 = self._get_skip_features(enc2_high, enc2_low)
        dec2 = torch.cat([dec2, skip2], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        skip1 = self._get_skip_features(enc1_high, enc1_low)
        dec1 = torch.cat([dec1, skip1], dim=1)
        dec1 = self.decoder1(dec1)

        # Final segmentation (returns logits, no sigmoid)
        output = self.conv_final(dec1)

        return output

    @staticmethod
    def _make_block(in_channels, out_channels, name):
        """Create a standard encoder/decoder block"""
        return nn.Sequential(
            OrderedDict([
                (f"{name}_conv1", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)),
                (f"{name}_norm1", nn.BatchNorm2d(out_channels)),
                (f"{name}_relu1", nn.ReLU(inplace=True)),
                (f"{name}_conv2", nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)),
                (f"{name}_norm2", nn.BatchNorm2d(out_channels)),
                (f"{name}_relu2", nn.ReLU(inplace=True)),
            ])
        )


# Example usage and testing
if __name__ == "__main__":
    batch_size = 4
    img_high = torch.randn(batch_size, 3, 128, 128)
    img_low = torch.randn(batch_size, 3, 128, 128)

    print("Testing LatentDiffUNet with different skip modes:\n")

    for skip_mode in ['high', 'low', 'both', 'avg']:
        model = LatentDiffUNet(in_channels=3, out_channels=2, init_features=32, skip_mode=skip_mode)
        output = model(img_high, img_low)

        n_params = sum(p.numel() for p in model.parameters())

        print(f"  skip_mode='{skip_mode}':")
        print(f"    Output shape: {output.shape}")
        print(f"    Parameters: {n_params:,}")
        print()

    print("✓ All skip modes work!")
