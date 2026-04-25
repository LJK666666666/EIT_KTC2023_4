"""VAE-style latent model for TD16 pulmonary delta-conductivity images."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dct_predictor.model import DCTDecoder


class ConvVAE(nn.Module):
    """Compact convolutional VAE for continuous 256x256 delta-sigma images."""

    def __init__(self, in_channels=1, latent_dim=32, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        channels = [base_channels, 64, 128, 256, 256, 256]
        encoder = []
        prev = in_channels
        for ch in channels:
            encoder.extend([
                nn.Conv2d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            prev = ch
        self.encoder = nn.Sequential(*encoder)
        self.enc_out_channels = channels[-1]
        self.enc_out_size = 4
        flat_dim = self.enc_out_channels * self.enc_out_size * self.enc_out_size
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flat_dim)
        decoder = []
        dec_channels = [256, 256, 128, 64, 32, 16]
        prev = self.enc_out_channels
        for ch in dec_channels:
            decoder.extend([
                nn.ConvTranspose2d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            prev = ch
        self.decoder = nn.Sequential(*decoder)
        self.out_conv = nn.Conv2d(prev, in_channels, kernel_size=3, padding=1)

    def encode(self, x):
        feats = self.encoder(x)
        feats = feats.reshape(x.shape[0], -1)
        mu = self.fc_mu(feats)
        logvar = self.fc_logvar(feats)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        feats = self.decoder_fc(z)
        feats = feats.view(z.shape[0], self.enc_out_channels, self.enc_out_size, self.enc_out_size)
        out = self.decoder(feats)
        out = self.out_conv(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        return 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)


class LatentMLPPredictor(nn.Module):
    """Simple MLP used to map 208-channel TD16 signals to VAE latent vectors."""

    def __init__(self, input_dim=208, latent_dim=32, hidden_dims=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for hid in hidden_dims:
            layers.extend([
                nn.Linear(prev, hid),
                nn.LayerNorm(hid),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev = hid
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, latent_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


class ConditionalLatentMLPPredictor(nn.Module):
    """Measurement-to-latent predictor with an explicit low-frequency mask branch."""

    def __init__(self,
                 input_dim=208,
                 latent_dim=32,
                 hidden_dims=(512, 256, 128),
                 dropout=0.1,
                 mask_coeff_size=24,
                 mask_condition_dim=128):
        super().__init__()
        self.mask_coeff_size = mask_coeff_size
        layers = []
        prev = input_dim
        for hid in hidden_dims:
            layers.extend([
                nn.Linear(prev, hid),
                nn.LayerNorm(hid),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev = hid
        self.backbone = nn.Sequential(*layers)
        self.mask_head = nn.Linear(prev, mask_coeff_size * mask_coeff_size)
        self.mask_decoder = DCTDecoder(
            image_size=256, coeff_size=mask_coeff_size, out_channels=1)
        self.mask_condition = nn.Sequential(
            nn.Linear(mask_coeff_size * mask_coeff_size, mask_condition_dim),
            nn.LayerNorm(mask_condition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(prev + mask_condition_dim, prev),
            nn.LayerNorm(prev),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prev, latent_dim),
        )

    def forward(self, x, mask_coeffs_override=None):
        feat = self.backbone(x)
        mask_coeffs_pred = self.mask_head(feat).view(
            x.shape[0], 1, self.mask_coeff_size, self.mask_coeff_size)
        cond_source = mask_coeffs_override
        if cond_source is None:
            cond_source = mask_coeffs_pred
        cond_feat = self.mask_condition(cond_source.reshape(x.shape[0], -1))
        latent = self.latent_head(torch.cat([feat, cond_feat], dim=1))
        mask_logits = self.mask_decoder.coeffs_to_logits(mask_coeffs_pred)
        return latent, mask_logits, mask_coeffs_pred

    def target_mask_coeffs(self, mask_target: torch.Tensor) -> torch.Tensor:
        return self.mask_decoder.images_to_coeffs(mask_target)
