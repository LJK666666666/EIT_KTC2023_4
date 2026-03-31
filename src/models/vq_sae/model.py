"""
Spatial-Transformer 1D Vector-Quantized VAE for EIT conductivity images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 5, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=4, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return F.normalize(self.head(feat), p=2, dim=-1)


class VQEncoderCNN(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 128, 256),
                 num_slots=16, code_dim=32):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1] * 16 * 16, num_slots * code_dim)
        self.num_slots = num_slots
        self.code_dim = code_dim

    def forward(self, x):
        feat = self.conv(x)
        z = self.fc(feat.flatten(1))
        return z.view(-1, self.num_slots, self.code_dim)


class VectorQuantizer1D(nn.Module):
    def __init__(self, codebook_size=512, code_dim=32, beta=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size,
                                            1.0 / codebook_size)

    def forward(self, z_e):
        bsz, num_slots, code_dim = z_e.shape
        flat = z_e.reshape(-1, code_dim).float()
        emb = self.embedding.weight.float()
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ emb.t()
            + emb.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(bsz, num_slots, code_dim)

        codebook_loss = F.mse_loss(z_q.float(), z_e.detach().float())
        commit_loss = F.mse_loss(z_e.float(), z_q.detach().float())
        vq_loss = codebook_loss + self.beta * commit_loss
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices.view(bsz, num_slots), vq_loss

    def lookup(self, indices):
        return self.embedding(indices.long())


class VQDecoder(nn.Module):
    def __init__(self, num_slots=16, code_dim=32, start_ch=256, start_size=4):
        super().__init__()
        self.fc = nn.Linear(num_slots * code_dim,
                            start_ch * start_size * start_size)
        self.start_ch = start_ch
        self.start_size = start_size
        self.up = nn.Sequential(
            self._up_block(256, 256),
            self._up_block(256, 128),
            self._up_block(128, 64),
            self._up_block(64, 32),
            self._up_block(32, 16),
            self._up_block(16, 8),
        )
        self.out_conv = nn.Conv2d(8, 3, 1)

    @staticmethod
    def _up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, z_q):
        x = self.fc(z_q.flatten(1))
        x = x.view(-1, self.start_ch, self.start_size, self.start_size)
        x = self.up(x)
        return self.out_conv(x)


def _build_rotation_matrix(theta):
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    row1 = torch.stack([cos_t, -sin_t, zeros], dim=1)
    row2 = torch.stack([sin_t, cos_t, zeros], dim=1)
    return torch.stack([row1, row2], dim=1)


class ST1DVQVAE(nn.Module):
    def __init__(self, in_channels=3, encoder_channels=(32, 64, 128, 256),
                 num_slots=16, codebook_size=512, code_dim=32,
                 decoder_start_size=4, vq_beta=0.25):
        super().__init__()
        self.num_slots = num_slots
        self.codebook_size = codebook_size
        self.code_dim = code_dim

        self.angle_cnn = AngleCNN(in_channels)
        self.encoder = VQEncoderCNN(
            in_channels=in_channels,
            channels=encoder_channels,
            num_slots=num_slots,
            code_dim=code_dim,
        )
        self.quantizer = VectorQuantizer1D(
            codebook_size=codebook_size,
            code_dim=code_dim,
            beta=vq_beta,
        )
        self.decoder = VQDecoder(
            num_slots=num_slots,
            code_dim=code_dim,
            start_ch=encoder_channels[-1],
            start_size=decoder_start_size,
        )

    def _stn_rotate(self, image, theta, inverse=False, logits=False):
        if inverse:
            theta = -theta
        affine = _build_rotation_matrix(theta)
        grid = F.affine_grid(affine, image.shape, align_corners=False)
        mode = 'bilinear' if (logits or self.training) else 'nearest'
        return F.grid_sample(
            image, grid, mode=mode, padding_mode='zeros',
            align_corners=False)

    def encode_continuous(self, x):
        angle_xy = self.angle_cnn(x)
        theta = torch.atan2(angle_xy[:, 1], angle_xy[:, 0])
        x_aligned = self._stn_rotate(x, theta, inverse=True, logits=False)
        z_e = self.encoder(x_aligned)
        return z_e, angle_xy

    def encode_indices(self, x):
        z_e, angle_xy = self.encode_continuous(x)
        _, indices, _ = self.quantizer(z_e)
        return indices, angle_xy

    def decode_quantized(self, z_q, angle_xy):
        canonical_logits = self.decoder(z_q)
        theta = torch.atan2(angle_xy[:, 1], angle_xy[:, 0])
        return self._stn_rotate(canonical_logits, theta, logits=True)

    def decode_from_indices(self, indices, angle_xy):
        z_q = self.quantizer.lookup(indices)
        return self.decode_quantized(z_q, angle_xy)

    def forward(self, x):
        z_e, angle_xy = self.encode_continuous(x)
        z_q, indices, vq_loss = self.quantizer(z_e)
        logits = self.decode_quantized(z_q, angle_xy)
        return logits, indices, angle_xy, vq_loss

