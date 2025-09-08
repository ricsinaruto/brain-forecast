import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..layers.attention import sdpa, sdpa_math
from ..layers.embeddings import rope_inv_freq, apply_rope_1d


def split_heads(x, heads):  # [B,C,T,H,W] -> [B,h,d,T,H,W]
    B, C, T, H, W = x.shape
    d = C // heads
    return x.view(B, heads, d, T, H, W)


def merge_heads(x):  # [B,h,d,T,H,W] -> [B,C,T,H,W]
    B, h, d, T, H, W = x.shape
    return x.view(B, h * d, T, H, W)


class CausalDWConv3d(nn.Module):
    """
    Depthwise Conv3d with left-only padding in time and 'same' padding in space.
    kernel_size: (k_t, k_h, k_w); stride must be 1 on time.
    """

    def __init__(self, channels, kernel_size=(3, 3, 3), bias=False):
        super().__init__()
        k_t, k_h, k_w = kernel_size
        assert k_t >= 1 and k_h >= 1 and k_w >= 1
        self.k_t, self.k_h, self.k_w = k_t, k_h, k_w
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=channels,
            bias=bias,
        )

    def forward(self, x):  # x: [B,C,T,H,W]
        k_t, k_h, k_w = self.k_t, self.k_h, self.k_w
        pad_t = (k_t - 1, 0)  # left-only in time
        pad_h = (k_h // 2, k_h // 2)  # symmetric in H
        pad_w = (k_w // 2, k_w // 2)  # symmetric in W
        # F.pad uses order: (wL,wR,hL,hR,dL,dR)
        x = F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_t[0], pad_t[1]))
        return self.conv(x)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, bias=True):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=0, bias=bias)

    def forward(self, x):  # x: [B,C,T]
        x = F.pad(x, (self.k - 1, 0))  # left-only
        return self.conv(x)


class CausalMixPredictor(nn.Module):
    """
    Produce per-time mixture weights α_m(t) using a causal conv over time.
    """

    def __init__(self, channels, M=3, k=3):
        super().__init__()
        self.pre = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.causal = CausalConv1d(channels, M, k=k)

    def forward(self, q):  # q: [B,C,T,H,W]
        # pool H/W but keep T so it's causal per-step
        qg = q.mean(dim=(3, 4))  # [B,C,T]
        h = self.pre(qg)
        h = self.act(h)
        logits = self.causal(h)  # [B,M,T]
        alpha = torch.softmax(logits, dim=1).to(q.dtype)  # softmax over kernel bank M
        return alpha


class ConvKernelBank(nn.Module):
    """
    Mixture-of-depthwise-3D kernels with strictly causal time:
      - K/V filtered by CausalDWConv3d (left-only time padding)
      - per-time mixture weights α_m(t) from CausalMixPredictor
    """

    def __init__(self, channels, M=3, k_t=3, k_spatial=3, mix_k=3):
        super().__init__()
        self.M = M
        ks = (k_t, k_spatial, k_spatial)
        self.kernels_k = nn.ModuleList(
            [CausalDWConv3d(channels, ks, bias=False) for _ in range(M)]
        )
        self.kernels_v = nn.ModuleList(
            [CausalDWConv3d(channels, ks, bias=False) for _ in range(M)]
        )
        self.mix = CausalMixPredictor(channels, M=M, k=mix_k)

    def forward(self, q, k, v):  # q,k,v: [B,C,T,H,W]
        B, C, T, H, W = k.shape
        # per-time mixture weights α_m(t)
        alpha = self.mix(q)  # [B,M,T]
        k_out = torch.zeros_like(k)
        v_out = torch.zeros_like(v)
        for m in range(self.M):
            km = self.kernels_k[m](k)
            vm = self.kernels_v[m](v)
            w = alpha[:, m].to(k.dtype).view(B, 1, T, 1, 1)
            k_out = k_out + w * km
            v_out = v_out + w * vm
        return k_out, v_out


class CK3DAttention(nn.Module):
    """
    - Tri-axial attention (T causal, H/W non-causal)
    - Depthwise 3D conv preconditioning on K,V
    - Per-axis RoPE (independent bases)
    """

    def __init__(
        self,
        channels,
        heads=8,
        kernel_bank_M=3,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rope=True,
        rope_base_t=10000.0,
        rope_base_h=10000.0,
        rope_base_w=10000.0,
    ):
        super().__init__()
        assert channels % heads == 0
        self.channels = channels
        self.heads = heads
        self.qkv = nn.Conv3d(channels, 3 * channels, 1, bias=False)
        self.bank = ConvKernelBank(channels, M=kernel_bank_M)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(channels, channels, 1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate = nn.Conv3d(channels, 3 * channels, 1)

        # RoPE
        self.use_rope = use_rope
        d_head = channels // heads
        d_rot = (d_head // 2) * 2
        self.register_buffer(
            "inv_freq_t", rope_inv_freq(d_rot, rope_base_t), persistent=False
        )
        self.register_buffer(
            "inv_freq_h", rope_inv_freq(d_rot, rope_base_h), persistent=False
        )
        self.register_buffer(
            "inv_freq_w", rope_inv_freq(d_rot, rope_base_w), persistent=False
        )

    def _attend_axis(self, q, k, v, axis: str, causal: bool):
        """
        q,k,v: [B, heads, d, T, H, W] -> returns [B, heads, d, T, H, W]
        """
        B, Hh, d, T, H, W = q.shape

        if axis == "t":
            qx = q.permute(0, 4, 5, 1, 3, 2).contiguous()
            qx = qx.view(B * H * W, Hh, T, d).contiguous()
            kx = k.permute(0, 4, 5, 1, 3, 2).contiguous()
            kx = kx.view(B * H * W, Hh, T, d).contiguous()
            vx = v.permute(0, 4, 5, 1, 3, 2).contiguous()
            vx = vx.view(B * H * W, Hh, T, d).contiguous()
            if self.use_rope and self.inv_freq_t.numel() > 0:
                qx, kx = apply_rope_1d(qx, kx, self.inv_freq_t, 0, L=T)
            out = sdpa(qx, kx, vx, causal=causal)
            out = out.view(B, H, W, Hh, T, d).permute(0, 3, 5, 4, 1, 2).contiguous()
            return out

        elif axis == "h":
            qx = q.permute(0, 3, 5, 1, 4, 2).contiguous()
            qx = qx.view(B * T * W, Hh, H, d).contiguous()
            kx = k.permute(0, 3, 5, 1, 4, 2).contiguous()
            kx = kx.view(B * T * W, Hh, H, d).contiguous()
            vx = v.permute(0, 3, 5, 1, 4, 2).contiguous()
            vx = vx.view(B * T * W, Hh, H, d).contiguous()
            if self.use_rope and self.inv_freq_h.numel() > 0:
                qx, kx = apply_rope_1d(qx, kx, self.inv_freq_h, 0, L=H)
            out = sdpa_math(qx, kx, vx)
            out = out.view(B, T, W, Hh, H, d).permute(0, 3, 5, 1, 4, 2).contiguous()
            return out

        elif axis == "w":
            qx = q.permute(0, 3, 4, 1, 5, 2).contiguous()
            qx = qx.view(B * T * H, Hh, W, d).contiguous()
            kx = k.permute(0, 3, 4, 1, 5, 2).contiguous()
            kx = kx.view(B * T * H, Hh, W, d).contiguous()
            vx = v.permute(0, 3, 4, 1, 5, 2).contiguous()
            vx = vx.view(B * T * H, Hh, W, d).contiguous()
            if self.use_rope and self.inv_freq_w.numel() > 0:
                qx, kx = apply_rope_1d(qx, kx, self.inv_freq_w, 0, L=W)
            out = sdpa_math(qx, kx, vx)
            out = out.view(B, T, H, Hh, W, d).permute(0, 3, 5, 1, 2, 4).contiguous()
            return out
        else:
            raise ValueError("axis ∈ {t,h,w}")

    def forward(self, x):  # [B,C,T,H,W]
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        k, v = self.bank(q, k, v)

        q = split_heads(q, self.heads)
        k = split_heads(k, self.heads)
        v = split_heads(v, self.heads)

        ctx_t = self._attend_axis(q, k, v, "t", causal=True)
        ctx_h = self._attend_axis(q, k, v, "h", causal=False)
        ctx_w = self._attend_axis(q, k, v, "w", causal=False)

        ctx_t = merge_heads(ctx_t)
        ctx_h = merge_heads(ctx_h)
        ctx_w = merge_heads(ctx_w)

        gt, gh, gw = torch.chunk(torch.sigmoid(self.gate(x)), 3, dim=1)
        y = gt * ctx_t + gh * ctx_h + gw * ctx_w
        y = self.attn_drop(y)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


class ConvMLP3D(nn.Module):
    """
    Pointwise -> CausalDWConv3d -> Pointwise. Fully causal in time.
    (Set k_t=1 if you prefer to avoid any temporal mixing here.)
    """

    def __init__(self, channels, expansion=4, drop=0.0, k_t=3, k_spatial=3):
        super().__init__()
        hidden = channels * expansion
        self.pw1 = nn.Conv3d(channels, hidden, 1)
        self.act1 = nn.GELU()
        self.dw = CausalDWConv3d(hidden, (k_t, k_spatial, k_spatial))
        self.act2 = nn.GELU()
        self.pw2 = nn.Conv3d(hidden, channels, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.pw1(x)
        x = self.act1(x)
        x = self.dw(x)
        x = self.act2(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x


class CK3DBlock(nn.Module):
    def __init__(self, channels, heads=8, mlp_expansion=4, drop=0.0, use_rope=True):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = CK3DAttention(
            channels,
            heads=heads,
            kernel_bank_M=3,
            attn_drop=drop,
            proj_drop=drop,
            use_rope=use_rope,
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.mlp = ConvMLP3D(channels, expansion=mlp_expansion, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CK3D(nn.Module):
    def __init__(
        self,
        emb_dim=16,
        quant_levels=256,
        channels=128,
        depth=6,
        heads=8,
        drop=0.0,
        use_rope=True,
    ):
        super().__init__()
        self.token = nn.Embedding(quant_levels, emb_dim)
        self.in_proj = nn.Conv3d(emb_dim, channels, 1)
        self.blocks = nn.ModuleList(
            [
                CK3DBlock(
                    channels, heads=heads, mlp_expansion=4, drop=drop, use_rope=use_rope
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.GroupNorm(1, channels)
        self.head = nn.Conv3d(channels, quant_levels, 1)  # categorical logits

    def forward(self, x_tokens: torch.Tensor, embeds: torch.Tensor = None):  # [B,H,W,T]
        if embeds is None:
            x = self.token(x_tokens)  # [B,H,W,T,D]
        else:
            x = embeds

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # [B,D,T,H,W]

        x = self.in_proj(x)  # [B,C,T,H,W]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x)  # [B,256,T,H,W]
        return logits.permute(0, 3, 4, 2, 1).contiguous()  # [B,H,W,T,256]


class Down2x(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv3d(
            C, C, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), groups=C
        )

    def forward(self, x):
        return self.conv(x)


class Up2x(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv3d(C, C, 1)

    def forward(self, x, size_t_hw):  # size_t_hw = (T, H, W)
        T, H, W = size_t_hw
        # Spatial-only upsample keeps temporal causality intact
        x = F.interpolate(
            x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
        )
        # (T stays identical;
        # using scale_factor avoids any accidental temporal resampling)
        return self.conv(x)


class CK3DVideoARPyramid(nn.Module):
    """
    Two-scale pyramid:
      - Full-res stream (depth_fine_1 + fusion + depth_fine_2)
      - Coarse spatial stream (H/2,W/2) with depth_coarse blocks,
      fused back by upsampling
    """

    def __init__(
        self,
        emb_dim=16,
        channels=128,
        quant_levels=256,
        depth_fine_1=3,
        depth_coarse=3,
        depth_fine_2=3,
        heads=8,
        drop=0.0,
        use_rope=True,
    ):
        super().__init__()
        self.token = nn.Embedding(quant_levels, emb_dim)
        self.in_proj = nn.Conv3d(emb_dim, channels, 1)

        self.fine1 = nn.ModuleList(
            [
                CK3DBlock(channels, heads=heads, drop=drop, use_rope=use_rope)
                for _ in range(depth_fine_1)
            ]
        )

        self.down = Down2x(channels)
        self.coarse = nn.ModuleList(
            [
                CK3DBlock(channels, heads=heads, drop=drop, use_rope=use_rope)
                for _ in range(depth_coarse)
            ]
        )
        self.up = Up2x(channels)

        self.fuse = nn.Conv3d(channels * 2, channels, 1)  # concat + 1x1 fuse
        self.fine2 = nn.ModuleList(
            [
                CK3DBlock(channels, heads=heads, drop=drop, use_rope=use_rope)
                for _ in range(depth_fine_2)
            ]
        )

        self.norm = nn.GroupNorm(1, channels)
        self.head = nn.Conv3d(channels, 256, 1)

    def forward(self, x_tokens):  # [B,T,H,W]
        x = self.in_proj(self.token(x_tokens))  # [B,C,T,H,W]
        for blk in self.fine1:
            x = blk(x)

        # Coarse spatial stream
        xc = self.down(x)
        for blk in self.coarse:
            xc = blk(xc)
        B, C, T, Hc, Wc = xc.shape
        xcu = self.up(xc, (T, Hc * 2, Wc * 2))  # back to full res

        # Fuse and refine
        x = torch.cat([x, xcu], dim=1)
        x = self.fuse(x)
        for blk in self.fine2:
            x = blk(x)

        x = self.norm(x)
        logits = self.head(x)  # [B,256,T,H,W]
        return logits.permute(0, 3, 4, 2, 1).contiguous()  # [B,H,W,T,256]


# ============================================================
# Discretized Logistic Mixture head (single-channel)
#   (PixelCNN++-style, no color coupling; K mixtures)
# ============================================================


class LogisticMixtureHead(nn.Module):
    def __init__(self, channels, K=10):
        super().__init__()
        self.K = K
        self.proj = nn.Conv3d(channels, 3 * K, 1)  # [pi_logits, means, log_scales]

    def forward(self, x):  # x: [B,C,T,H,W] -> params: [B,3K,T,H,W]
        return self.proj(x)


def mixture_logistic_nll(params, targets):
    """
    params: [B, 3K, T, H, W] from LogisticMixtureHead
    targets: int pixels in [0,255], shape [B, T, H, W]
    Return: mean NLL over all sites.
    """
    B, C3K, T, H, W = params.shape
    K = C3K // 3
    logits, means, log_scales = torch.split(params, K, dim=1)  # each [B,K,T,H,W]

    # targets to [-1,1]
    x = targets.float() / 255.0 * 2.0 - 1.0  # [B,T,H,W]
    x = x.unsqueeze(1)  # [B,1,T,H,W]
    means = means
    log_scales = torch.clamp(log_scales, min=-7.0)  # avoid extremely small scales
    inv_scales = torch.exp(-log_scales)

    # bin size in [-1,1] range for 256 levels
    bin_size = 2.0 / 255.0

    plus = torch.sigmoid((x + bin_size / 2 - means) * inv_scales)
    minus = torch.sigmoid((x - bin_size / 2 - means) * inv_scales)
    # probability mass in the discrete bin
    probs = torch.clamp(plus - minus, min=1e-12)  # [B,K,T,H,W]

    log_mix = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # [B,K,T,H,W]
    log_prob = torch.log(probs) + log_mix  # [B,K,T,H,W]
    log_prob = torch.logsumexp(log_prob, dim=1)  # [B,T,H,W]
    nll = -log_prob.mean()
    return nll


@torch.no_grad()
def mixture_logistic_sample(params, temperature=1.0):
    """
    params: [B,3K,T,H,W] -> sampled integer pixels [B,T,H,W] in [0,255]
    """
    B, C3K, T, H, W = params.shape
    K = C3K // 3
    logits, means, log_scales = torch.split(params, K, dim=1)
    # sample mixture index
    mix = torch.distributions.Categorical(
        logits=logits / max(1e-6, float(temperature))
    ).sample()  # [B,T,H,W]
    m_idx = mix.unsqueeze(1)  # [B,1,T,H,W]
    means_sel = torch.gather(means, 1, m_idx)  # [B,1,T,H,W]
    log_scales_sel = torch.gather(log_scales, 1, m_idx)  # [B,1,T,H,W]
    scales_sel = torch.exp(log_scales_sel)

    # sample from logistic
    u = torch.clamp(torch.rand_like(means_sel), 1e-5, 1 - 1e-5)
    x = means_sel + scales_sel * torch.log(u / (1 - u))  # logistic sample
    # to [0,255]
    x = (
        torch.clamp((x.squeeze(1) + 1.0) * 0.5 * 255.0, 0, 255).round().long()
    )  # [B,T,H,W]
    return x


# ============================================================
# Mixture model variant (same backbone, different head & loss)
# ============================================================


class CK3DVideoARMixture(nn.Module):
    def __init__(
        self,
        emb_dim=16,
        quant_levels=256,
        channels=128,
        depth=6,
        heads=8,
        drop=0.0,
        K=10,
        use_rope=True,
    ):
        super().__init__()
        self.token = nn.Embedding(quant_levels, emb_dim)
        self.in_proj = nn.Conv3d(emb_dim, channels, 1)
        self.blocks = nn.ModuleList(
            [
                CK3DBlock(channels, heads=heads, drop=drop, use_rope=use_rope)
                for _ in range(depth)
            ]
        )
        self.norm = nn.GroupNorm(1, channels)
        self.head = LogisticMixtureHead(channels, K=K)
        self.K = K

    def forward(
        self, x_tokens
    ):  # [B,T,H,W] -> params [B,T,H,W,3K] (for consistency with your API)
        x = self.in_proj(self.token(x_tokens))  # [B,C,T,H,W]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        params = self.head(x)  # [B,3K,T,H,W]
        return params.permute(0, 3, 4, 2, 1).contiguous()  # [B,H,W,T,3K]

    # convenience helpers
    def nll(self, x_tokens, targets):
        params = self.head(self._features(x_tokens))
        return mixture_logistic_nll(params, targets)

    def _features(self, x_tokens):
        x = self.in_proj(self.token(x_tokens))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# ============================================================
# Minimal toy dataset + tiny train loops (both heads)
# ============================================================


class MovingSquare(Dataset):
    def __init__(self, num_samples=1024, T=16, H=32, W=32, side=6):
        super().__init__()
        self.N = num_samples
        self.T = T
        self.H = H
        self.W = W
        self.side = side

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        H, W, T, s = self.H, self.W, self.T, self.side
        x = torch.zeros(T, H, W, dtype=torch.long)
        r = torch.randint(0, H - s, (1,)).item()
        c = torch.randint(0, W - s, (1,)).item()
        vr = torch.randint(-1, 2, (1,)).item() or 1
        vc = torch.randint(-1, 2, (1,)).item() or 1
        for t in range(T):
            x[t, r : r + s, c : c + s] = 255
            rn, cn = r + vr, c + vc
            if rn < 0 or rn + s > H:
                vr = -vr
            if cn < 0 or cn + s > W:
                vc = -vc
            r += vr
            c += vc
        inp, tgt = x[:-1], x[1:]  # teacher-forcing
        return inp, tgt


def train_categorical(epochs=2, batch_size=16, T=16, H=32, W=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = MovingSquare(num_samples=512, T=T, H=H, W=W)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = CK3DVideoARPyramid(
        emb_dim=16,
        channels=128,
        depth_fine_1=2,
        depth_coarse=2,
        depth_fine_2=2,
        heads=8,
        drop=0.1,
        use_rope=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for inp, tgt in dl:
            inp, tgt = inp.to(device), tgt.to(device)  # [B,T-1,H,W]
            logits = model(inp)  # [B,H,W,T-1,256]
            B, Hh, Ww, Tm1, Q = logits.shape
            loss = F.cross_entropy(
                logits.permute(0, 3, 1, 2, 4).contiguous().view(B * Tm1 * Hh * Ww, Q),
                tgt.view(B * Tm1 * Hh * Ww),
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        print(f"[Cat] Epoch {ep}: loss={total/len(dl):.4f}")
    return model


def train_mixture(epochs=2, batch_size=16, T=16, H=32, W=32, K=10, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = MovingSquare(num_samples=512, T=T, H=H, W=W)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = CK3DVideoARMixture(
        emb_dim=16, channels=128, depth=6, heads=8, drop=0.1, K=K, use_rope=True
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for inp, tgt in dl:
            inp, tgt = inp.to(device), tgt.to(device)  # [B,T-1,H,W]
            # forward features and head
            x = model.in_proj(model.token(inp))
            for blk in model.blocks:
                x = blk(x)
            x = model.norm(x)
            params = model.head(x)  # [B,3K,T-1,H,W]
            loss = mixture_logistic_nll(params, tgt)  # scalar
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        print(f"[Mix] Epoch {ep}: nll={total/len(dl):.4f}")
    return model


# ============================================================
# Quick smoke test
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cat_model = train_categorical(epochs=2, device=device)
    mix_model = train_mixture(epochs=2, K=5, device=device)

    # sampling examples
    with torch.no_grad():
        B, T0, H, W = 1, 8, 32, 32
        seed = torch.randint(0, 256, (B, T0, H, W), dtype=torch.long, device=device)

        # categorical sampling (autoregressive over time)
        logits = cat_model(seed)  # [B,H,W,T0,256]
        last = logits[..., -1, :] / 1.0
        probs = F.softmax(last.view(B, H * W, 256), dim=-1)
        next_tok = (
            torch.multinomial(probs, 1).squeeze(-1).view(B, H, W).unsqueeze(1)
        )  # [B,1,H,W]
        print("Cat next frame shape:", next_tok.shape)

        # mixture sampling
        x = mix_model.in_proj(mix_model.token(seed))
        for blk in mix_model.blocks:
            x = blk(x)
        x = mix_model.norm(x)
        params = mix_model.head(x)  # [B,3K,T0,H,W]
        nxt = mixture_logistic_sample(
            params[..., -1:, :, :]
        )  # sample just next frame from last-step params
        print("Mix next frame shape:", nxt.permute(0, 2, 3, 1).shape)  # [B,H,W,1]
