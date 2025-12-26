"""
Latent Difference U-Net
Novel architecture: Separate encoders → Latent difference → Shared decoder

High → Encoder1 ─┐
                  ├→ Difference (in latent space) → Shared Decoder → Segmentation
Low  → Encoder2 ─┘
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class LatentDiffUNet(nn.Module):
    """
    Dual-encoder architecture with latent space difference fusion.

    Unlike Semi-Siamese (which differences decoder outputs),
    this model differences encoder outputs in the latent space.

    Args:
        in_channels: Input image channels (default 3)
        out_channels: Output segmentation classes (default 2)
        init_features: Initial feature channels (default 32)
    """

    def __init__(self, in_channels=3, out_channels=2, init_features=32):
        super(LatentDiffUNet, self).__init__()

        features = init_features

        # Separate encoders for high and low fidelity
        self.encoder1_high = self._make_encoder_block(in_channels, features, "high_enc1")
        self.pool1_high = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_high = self._make_encoder_block(features, features * 2, "high_enc2")
        self.pool2_high = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_high = self._make_encoder_block(features * 2, features * 4, "high_enc3")
        self.pool3_high = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_high = self._make_encoder_block(features * 4, features * 8, "high_enc4")
        self.pool4_high = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1_low = self._make_encoder_block(in_channels, features, "low_enc1")
        self.pool1_low = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_low = self._make_encoder_block(features, features * 2, "low_enc2")
        self.pool2_low = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_low = self._make_encoder_block(features * 2, features * 4, "low_enc3")
        self.pool3_low = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_low = self._make_encoder_block(features * 4, features * 8, "low_enc4")
        self.pool4_low = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck processes the difference
        self.bottleneck = self._make_encoder_block(features * 8, features * 16, "bottleneck")

        # Shared decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._make_encoder_block((features * 8) * 2, features * 8, "dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._make_encoder_block((features * 4) * 2, features * 4, "dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._make_encoder_block((features * 2) * 2, features * 2, "dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._make_encoder_block(features * 2, features, "dec1")

        # Final segmentation head
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, img_high, img_low):
        """
        Forward pass with latent space difference.

        Args:
            img_high: High-fidelity image [B, 3, H, W]
            img_low: Low-fidelity image [B, 3, H, W]

        Returns:
            Segmentation logits [B, out_channels, H, W]
        """
        # Encode high-fidelity
        enc1_high = self.encoder1_high(img_high)
        enc2_high = self.encoder2_high(self.pool1_high(enc1_high))
        enc3_high = self.encoder3_high(self.pool2_high(enc2_high))
        enc4_high = self.encoder4_high(self.pool3_high(enc3_high))
        pooled_high = self.pool4_high(enc4_high)

        # Encode low-fidelity
        enc1_low = self.encoder1_low(img_low)
        enc2_low = self.encoder2_low(self.pool1_low(enc1_low))
        enc3_low = self.encoder3_low(self.pool2_low(enc2_low))
        enc4_low = self.encoder4_low(self.pool3_low(enc3_low))
        pooled_low = self.pool4_low(enc4_low)

        # Compute difference in latent space
        latent_diff = pooled_high - pooled_low

        # Bottleneck processes the difference
        bottleneck = self.bottleneck(latent_diff)

        # Shared decoder with skip connections from high-fidelity encoder
        # (Using high-fidelity skips since that's the "ground truth")
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4_high), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3_high), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2_high), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1_high), dim=1)
        dec1 = self.decoder1(dec1)

        # Final segmentation (returns logits, no sigmoid)
        output = self.conv_final(dec1)

        return output

    @staticmethod
    def _make_encoder_block(in_channels, out_channels, name):
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


# Example usage
if __name__ == "__main__":
    # Test model
    batch_size = 4
    img_high = torch.randn(batch_size, 3, 128, 128)
    img_low = torch.randn(batch_size, 3, 128, 128)

    model = LatentDiffUNet(in_channels=3, out_channels=2, init_features=32)

    output = model(img_high, img_low)

    print("LatentDiffUNet Test:")
    print(f"  Input: high {img_high.shape}, low {img_low.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✓ Model works!")
