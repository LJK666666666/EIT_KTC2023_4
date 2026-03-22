"""
Harmonic Cascaded DPCA-UNet (HC-DPCA-UNet).

Architecture:
  Module 1: Harmonic Electrode Encoder
    - 8-order Fourier harmonics on electrode angles (NeRF-style)
    - Dual-track: voltage Linear(76,d) + coord MLP(16→64→d), summed
    - Self-attention among 31 electrodes
    - Separate K, V projections

  Module 2: Harmonic Spatial Query Generator
    - (x, y, r) + 8-order harmonics on x and y → 35-dim input
    - Deep MLP: 35 → d → d → d

  Module 3: Cascaded Cross-Attention (2-layer)
    - Two rounds of cross-attention with LayerNorm + FFN + residual

  Module 4: Information Bottleneck + Dual-Pool UNet
    - 1x1 conv compresses d_model → bottleneck_ch (32)
    - Encoder: DualPool + ConvBlock at each level
    - Decoder: ConvTranspose2d + skip concat + ConvBlock
    - Output head: 1x1 conv → 3 classes
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Electrode angle constants
# ---------------------------------------------------------------

def _compute_electrode_angles(n_electrodes=32):
    """Electrode center angles in radians (standard math convention).

    Electrode 1 at top (90 deg), counterclockwise, 11.25 deg spacing.
    """
    angles_deg = 90.0 + np.arange(n_electrodes) * (360.0 / n_electrodes)
    angles_rad = np.deg2rad(angles_deg)
    mid_angles = np.empty(n_electrodes - 1)
    for i in range(n_electrodes - 1):
        a1, a2 = angles_rad[i], angles_rad[i + 1]
        mid_angles[i] = a1 + (a2 - a1) / 2
    return angles_rad, mid_angles


def _fourier_features(angles, L=8):
    """NeRF-style Fourier positional encoding.

    Args:
        angles: (...,) array in radians.
        L: number of harmonic orders.

    Returns:
        (..., 2*L) array of [sin(θ), cos(θ), sin(2θ), cos(2θ), ...].
    """
    freqs = np.arange(1, L + 1)  # (L,)
    # (...,1) * (L,) → (...,L)
    scaled = angles[..., None] * freqs
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=-1)


# ---------------------------------------------------------------
# Module 1: Harmonic Electrode Encoder
# ---------------------------------------------------------------

class HarmonicElectrodeEncoder(nn.Module):
    """Encode measurements + harmonic electrode positions into K, V.

    Dual-track fusion:
      - Voltage track: Linear(n_patterns, d_model)
      - Coord track: MLP(2*L, 64, d_model) with GELU
      - Sum → self-attention → K, V projections
    """

    def __init__(self, n_patterns=76, d_model=128, n_channels=31,
                 n_heads=4, L=8):
        super().__init__()
        self.L = L
        coord_dim = 2 * L  # sin + cos for each order

        # Dual-track projections
        self.voltage_proj = nn.Linear(n_patterns, d_model)
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )

        # Self-attention among electrodes
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.0,
            activation='gelu', batch_first=True,
            norm_first=True)

        # K, V projections
        self.key_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)

    def forward(self, voltages, coord_features):
        """
        Args:
            voltages: (B, 31, 76) raw voltage measurements.
            coord_features: (31, 2*L) pre-computed harmonic features (buffer).

        Returns:
            K: (B, 31, d_model), V: (B, 31, d_model)
        """
        v_feat = self.voltage_proj(voltages)          # (B, 31, d)
        c_feat = self.coord_mlp(coord_features)       # (31, d)
        x = v_feat + c_feat.unsqueeze(0)              # (B, 31, d)

        x = self.self_attn(x)                         # (B, 31, d)

        return self.key_proj(x), self.val_proj(x)


# ---------------------------------------------------------------
# Module 2: Harmonic Spatial Query Generator
# ---------------------------------------------------------------

class HarmonicSpatialQueryMLP(nn.Module):
    """Generate spatial queries with full-spectrum harmonic features.

    Input: (x, y, r) + 8-order harmonics on x and y → 35 dims.
    MLP: 35 → d → d → d.
    """

    def __init__(self, d_model=128, im_size=256, L=8):
        super().__init__()
        self.im_size = im_size
        input_dim = 3 + 2 * L + 2 * L  # x,y,r + harmonics_x + harmonics_y = 35

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Pre-compute position encoding
        pix_width = 0.23 / im_size
        coords = np.linspace(
            -0.115 + pix_width / 2,
            0.115 - pix_width / 2 + pix_width, im_size)
        gx, gy = np.meshgrid(coords, coords, indexing='ij')
        scale = 0.098
        x_norm = gx / scale
        y_norm = gy / scale
        r_norm = np.sqrt(x_norm ** 2 + y_norm ** 2)

        # 8-order harmonics on x and y
        harm_x = _fourier_features(x_norm.ravel(), L)   # (H*W, 2L)
        harm_y = _fourier_features(y_norm.ravel(), L)   # (H*W, 2L)

        pos = np.concatenate([
            x_norm.ravel()[:, None],
            y_norm.ravel()[:, None],
            r_norm.ravel()[:, None],
            harm_x, harm_y,
        ], axis=-1).astype(np.float32)  # (H*W, 35)

        self.register_buffer('pos_encoding', torch.from_numpy(pos))

    def forward(self, batch_size):
        """Returns Q: (B, H*W, d_model)."""
        q = self.mlp(self.pos_encoding)
        return q.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------
# Module 3: Cascaded Cross-Attention
# ---------------------------------------------------------------

class CascadedCrossAttentionLayer(nn.Module):
    """One layer of cross-attention + FFN with pre-norm and residual."""

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_head = d_model // n_heads
        self.scale = math.sqrt(d_head)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def _cross_attention(self, Q, K, V, mask=None):
        B, Nq, D = Q.shape
        Nk = K.shape[1]
        h = self.n_heads
        d_h = D // h

        Q = Q.view(B, Nq, h, d_h).transpose(1, 2)
        K = K.view(B, Nk, h, d_h).transpose(1, 2)
        V = V.view(B, Nk, h, d_h).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.out_proj(out)

    def forward(self, Q, K, V, mask=None):
        # Pre-norm cross-attention + residual
        Q_norm = self.norm1(Q)
        K_norm = self.norm_k(K)
        Q = Q + self._cross_attention(Q_norm, K_norm, V, mask)
        # Pre-norm FFN + residual
        Q = Q + self.ffn(self.norm2(Q))
        return Q


class CascadedCrossAttention(nn.Module):
    """Two-layer cascaded cross-attention."""

    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            CascadedCrossAttentionLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, Q, K, V, mask=None):
        for layer in self.layers:
            Q = layer(Q, K, V, mask)
        return Q


# ---------------------------------------------------------------
# Module 4: Dual-Pool UNet building blocks (reused from DPCAUNet)
# ---------------------------------------------------------------

class DualPool(nn.Module):
    """MaxPool + MinPool -> channel concatenation (doubles channels)."""

    def __init__(self, kernel_size=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        return torch.cat([self.pool(x), -self.pool(-x)], dim=1)


class ConvBlock(nn.Module):
    """Two convolutions with BatchNorm and GELU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """DualPool -> ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dual_pool = DualPool()
        self.conv = ConvBlock(in_ch * 2, out_ch)

    def forward(self, x):
        return self.conv(self.dual_pool(x))


class DecoderBlock(nn.Module):
    """Upsample -> cat skip -> ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------
# Main model
# ---------------------------------------------------------------

class HCDPCAUNet(nn.Module):
    """Harmonic Cascaded DPCA-UNet for EIT image reconstruction.

    Args:
        n_channels: Number of differential channels (31).
        n_patterns: Number of excitation patterns (76).
        d_model: Hidden dimension for attention modules (128).
        n_heads: Number of attention heads (4).
        im_size: Output image size (256).
        bottleneck_ch: Channel width after information bottleneck (32).
        encoder_channels: UNet encoder channel widths.
        out_channels: Number of output classes (3).
        max_period: Timestep embedding frequency.
        harmonic_L: Number of Fourier harmonic orders (8).
        n_cascade_layers: Number of cascaded cross-attention layers (2).
    """

    def __init__(self, n_channels=31, n_patterns=76, d_model=128,
                 n_heads=4, im_size=256, bottleneck_ch=32,
                 encoder_channels=(32, 64, 128, 256), out_channels=3,
                 max_period=0.25, harmonic_L=8, n_cascade_layers=2):
        super().__init__()
        self.n_channels = n_channels
        self.n_patterns = n_patterns
        self.d_model = d_model
        self.im_size = im_size
        self.max_period = max_period

        # Pre-compute electrode harmonic features
        _, mid_angles = _compute_electrode_angles(n_channels + 1)
        elec_harmonics = _fourier_features(mid_angles, harmonic_L)  # (31, 2L)
        self.register_buffer(
            'electrode_harmonics',
            torch.from_numpy(elec_harmonics.astype(np.float32)))

        # Module 1: Harmonic Electrode Encoder
        self.electrode_encoder = HarmonicElectrodeEncoder(
            n_patterns=n_patterns, d_model=d_model,
            n_channels=n_channels, n_heads=n_heads, L=harmonic_L)

        # Module 2: Harmonic Spatial Query Generator
        self.spatial_query = HarmonicSpatialQueryMLP(
            d_model=d_model, im_size=im_size, L=harmonic_L)

        # Module 3: Cascaded Cross-Attention
        self.cascaded_attn = CascadedCrossAttention(
            d_model=d_model, n_heads=n_heads, n_layers=n_cascade_layers)

        # Level embedding
        self.level_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Module 4: Information bottleneck + UNet
        # 4a. 1x1 bottleneck: d_model -> bottleneck_ch
        self.bottleneck_conv = nn.Conv2d(d_model, bottleneck_ch, 1)

        # 4b. Initial ConvBlock
        self.initial_conv = ConvBlock(bottleneck_ch, encoder_channels[0])

        # 4c. Encoder (dual-pooling)
        self.encoders = nn.ModuleList()
        enc_in = [encoder_channels[0]] + list(encoder_channels[:-1])
        for c_in, c_out in zip(enc_in, encoder_channels):
            self.encoders.append(EncoderBlock(c_in, c_out))

        # 4d. Bottleneck conv
        self.unet_bottleneck = ConvBlock(encoder_channels[-1],
                                         encoder_channels[-1])

        # 4e. Decoder
        self.decoders = nn.ModuleList()
        dec_channels = list(reversed(encoder_channels))
        for i in range(len(dec_channels) - 1):
            self.decoders.append(
                DecoderBlock(dec_channels[i], dec_channels[i + 1],
                             dec_channels[i + 1]))
        self.decoders.append(
            DecoderBlock(dec_channels[-1], encoder_channels[0],
                         encoder_channels[0]))

        # 4f. Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0], out_channels, 1),
        )

        # Stage-1 pretrain head: 1x1 from d_model -> out_channels
        self.pretrain_head = nn.Conv2d(d_model, out_channels, 1)

        # Auxiliary heads for deep supervision (one per decoder block)
        self.aux_heads = nn.ModuleList()
        aux_ch_list = list(reversed(encoder_channels))[1:] + \
            [encoder_channels[0]]
        for ch in aux_ch_list:
            self.aux_heads.append(nn.Conv2d(ch, out_channels, 1))

    def _timestep_embedding(self, timesteps, dim):
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(
                half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def _build_channel_mask(self, measurements):
        """Detect active channels from vincl-masked measurements."""
        x = measurements.view(-1, self.n_channels, self.n_patterns)
        return x.abs().sum(dim=-1) > 0  # (B, 31)

    def _attention_forward(self, measurements, level):
        """Shared attention pipeline, returns feature map (B, d, H, W)."""
        B = measurements.shape[0]

        # Module 1: electrode encoding
        voltages = measurements.view(B, self.n_channels, self.n_patterns)
        K, V = self.electrode_encoder(voltages, self.electrode_harmonics)

        # Channel mask
        channel_mask = self._build_channel_mask(measurements)

        # Module 2: spatial queries
        Q = self.spatial_query(B)

        # Module 3: cascaded cross-attention
        feat = self.cascaded_attn(Q, K, V, mask=channel_mask)
        feat = feat.view(B, self.im_size, self.im_size,
                         self.d_model).permute(0, 3, 1, 2)

        # Level embedding (additive)
        level_emb = self._timestep_embedding(level, self.d_model)
        level_emb = self.level_embed(level_emb)
        feat = feat + level_emb[:, :, None, None]

        return feat

    def forward_pretrain(self, measurements, level):
        """Stage 1: attention + lightweight 1x1 head only."""
        feat = self._attention_forward(measurements, level)
        return self.pretrain_head(feat)

    def forward(self, measurements, level, deep_supervision=False):
        """
        Args:
            measurements: (B, 2356) flattened voltage differences.
            level: (B,) difficulty level (1-7).
            deep_supervision: If True, also return aux outputs.

        Returns:
            If deep_supervision=False: (B, 3, 256, 256) main logits.
            If deep_supervision=True:  (main_logits, [aux1, aux2, ...])
        """
        feat = self._attention_forward(measurements, level)

        # Information bottleneck: d_model -> bottleneck_ch
        feat = self.bottleneck_conv(feat)

        # UNet
        x0 = self.initial_conv(feat)

        skips = [x0]
        h = x0
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        h = self.unet_bottleneck(h)

        skips = skips[:-1]
        aux_outputs = []
        for i, dec in enumerate(self.decoders):
            h = dec(h, skips.pop())
            if deep_supervision:
                aux_logits = self.aux_heads[i](h)
                if aux_logits.shape[2:] != (self.im_size, self.im_size):
                    aux_logits = F.interpolate(
                        aux_logits, (self.im_size, self.im_size),
                        mode='bilinear', align_corners=False)
                aux_outputs.append(aux_logits)

        main_out = self.output_head(h)

        if deep_supervision:
            return main_out, aux_outputs
        return main_out
