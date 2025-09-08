import torch

from ephys_gpt.models.tokenizers.emu3visionvq import Emu3VisionVQ


def test_emu3visionvq_accepts_chunkdatasetimage_tensor():
    cfg = dict(
        in_channels=1,
        out_channels=1,
        temporal_downsample_factor=2,
        ch=64,
        ch_mult=[1, 2],
        num_res_blocks=1,
        z_channels=8,
        embed_dim=8,
        codebook_size=256,
        attn_resolutions=[],
    )
    model = Emu3VisionVQ(**cfg)

    # Simulate ChunkDatasetImage output: (H,W,T)
    H = W = 32
    T = 8
    img = torch.randn(H, W, T)

    z_q, _, codes = model.encode(img)

    # reshape codes to (B, T, H, W)
    codes = codes.reshape(z_q.shape[0], z_q.shape[1], z_q.shape[3], z_q.shape[4])

    # Decode back to (B,H,W,T)
    rec = model.decode(codes[0])  # pass (T',H',W')
    assert rec.shape == (1, H, W, T)
