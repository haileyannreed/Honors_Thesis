"""
Flexible U-Net for Multi-Fidelity Input
Supports both concatenation and difference fusion strategies
"""

import torch
import torch.nn as nn
from models.unet import UNet


class FlexibleUNet(nn.Module):
    """
    U-Net that can handle high-fidelity and low-fidelity inputs with different fusion strategies.

    Fusion modes:
    - 'concat': Concatenate [high, low] → 6 channels input
    - 'diff': Compute (high - low) → 3 channels input

    Args:
        fusion_mode: 'concat' or 'diff'
        out_channels: Number of output classes (default 2 for binary segmentation)
        init_features: Initial number of features in U-Net (default 32)
    """

    def __init__(self, fusion_mode='concat', out_channels=2, init_features=32):
        super(FlexibleUNet, self).__init__()

        assert fusion_mode in ['concat', 'diff'], "fusion_mode must be 'concat' or 'diff'"
        self.fusion_mode = fusion_mode

        # Determine input channels based on fusion mode
        in_channels = 6 if fusion_mode == 'concat' else 3

        # U-Net backbone
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features
        )

        # Remove sigmoid from UNet output (we need raw logits for loss)
        # Modify the final conv to not apply sigmoid
        self._modify_unet_output()

    def _modify_unet_output(self):
        """Remove sigmoid activation from U-Net output"""
        # The UNet's forward already applies sigmoid, so we'll override it
        # by wrapping the forward method
        original_forward = self.unet.forward

        def forward_without_sigmoid(x):
            # Copy the UNet forward but without sigmoid
            enc1 = self.unet.encoder1(x)
            enc2 = self.unet.encoder2(self.unet.pool1(enc1))
            enc3 = self.unet.encoder3(self.unet.pool2(enc2))
            enc4 = self.unet.encoder4(self.unet.pool3(enc3))

            bottleneck = self.unet.bottleneck(self.unet.pool4(enc4))

            dec4 = self.unet.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.unet.decoder4(dec4)
            dec3 = self.unet.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.unet.decoder3(dec3)
            dec2 = self.unet.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.unet.decoder2(dec2)
            dec1 = self.unet.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.unet.decoder1(dec1)

            # Return logits WITHOUT sigmoid
            return self.unet.conv(dec1)

        self.unet.forward = forward_without_sigmoid

    def forward(self, img_high, img_low):
        """
        Forward pass with multi-fidelity inputs.

        Args:
            img_high: High-fidelity image [B, 3, H, W]
            img_low: Low-fidelity image [B, 3, H, W]

        Returns:
            Segmentation logits [B, out_channels, H, W]
        """
        if self.fusion_mode == 'concat':
            # Concatenate along channel dimension
            x = torch.cat([img_high, img_low], dim=1)  # [B, 6, H, W]
        elif self.fusion_mode == 'diff':
            # Compute difference
            x = img_high - img_low  # [B, 3, H, W]

        # Pass through U-Net
        output = self.unet(x)

        return output


# Convenience wrappers for clarity
class ConcatUNet(FlexibleUNet):
    """Early Concatenation U-Net: [high, low] → U-Net"""
    def __init__(self, out_channels=2, init_features=32):
        super().__init__(fusion_mode='concat', out_channels=out_channels, init_features=init_features)


class DiffUNet(FlexibleUNet):
    """Difference U-Net: (high - low) → U-Net"""
    def __init__(self, out_channels=2, init_features=32):
        super().__init__(fusion_mode='diff', out_channels=out_channels, init_features=init_features)


# Example usage
if __name__ == "__main__":
    # Test both models
    batch_size = 4
    img_high = torch.randn(batch_size, 3, 128, 128)
    img_low = torch.randn(batch_size, 3, 128, 128)

    print("Testing ConcatUNet:")
    concat_model = ConcatUNet(out_channels=2)
    concat_out = concat_model(img_high, img_low)
    print(f"  Input: high {img_high.shape}, low {img_low.shape}")
    print(f"  Output: {concat_out.shape}")

    print("\nTesting DiffUNet:")
    diff_model = DiffUNet(out_channels=2)
    diff_out = diff_model(img_high, img_low)
    print(f"  Input: high {img_high.shape}, low {img_low.shape}")
    print(f"  Output: {diff_out.shape}")

    print("\n✓ Both models work!")
