import numpy as np
from typing import Any
import torch
from torch import Tensor
from sklearn.cluster import MiniBatchKMeans


def mulaw_torch(x: torch.Tensor, mu: int = 255) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Torch version of mu-law companding and quantization.

    Args:
        x: Input tensor (float, expected in [-1, 1])
        mu: Mu parameter (number of quantization levels - 1)

    Returns:
        Tuple of (quantized tensor, reconstructed tensor)
    """
    shape = x.shape
    x_flat = x.reshape(-1)

    # clip to -1, 1
    x_flat = torch.clamp(x_flat, -0.999, 0.999)

    # Mu-law compression
    compressed = (
        torch.sign(x_flat)
        * torch.log1p(mu * torch.abs(x_flat))
        / torch.log1p(torch.tensor(float(mu), device=x.device))
    )

    # Quantize to integers in [0, mu]
    compressed = (compressed + 1) * 0.5 * mu + 0.5
    digitized = compressed.long()

    return digitized.reshape(shape)


def mulaw(x: np.ndarray, mu: int = 255) -> tuple[np.ndarray, np.ndarray]:
    # Select smallest uint type that can hold mu value
    if mu <= 255:
        dtype = np.uint8
    elif mu <= 65535:
        dtype = np.uint16
    elif mu <= 4294967295:
        dtype = np.uint32
    else:
        dtype = np.uint64

    # Store original shape but work with flattened array
    shape = x.shape
    x_flat = x.ravel()

    # Compute mu-law compression in one step
    compressed = np.sign(x_flat) * np.log1p(mu * np.abs(x_flat)) / np.log1p(mu)

    # Quantize to integers in [0, mu]
    compressed = (compressed + 1) * 0.5 * mu + 0.5
    digitized = compressed.astype(dtype)

    # perform inverse mu-law compression
    x_recon = mulaw_inv(digitized, mu)

    return digitized.reshape(shape), x_recon.reshape(shape)


def mulaw_inv(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """
    Inverse mu-law companding.
    """
    # Scale from [0, mu] to [-1, 1]
    x_scaled = (x.astype(np.float32)) / mu * 2 - 1

    # Inverse mu-law transformation
    x_out = np.sign(x_scaled) * (1 / mu) * ((1 + mu) ** np.abs(x_scaled) - 1)

    return x_out


def mulaw_inv_torch(x_int: Tensor, mu: int = 255) -> Tensor:
    """Inverse µ‑law for integer bins ∈ [0, mu]. Returns float tensor ∈ [‑1, 1]."""
    mu = float(mu)
    x = x_int.float()
    x_norm = x / mu  # [0,1]
    x_norm = 2 * x_norm - 1  # [‑1,1]
    mag = (1 / mu) * ((1 + mu) ** x_norm.abs() - 1)
    return torch.sign(x_norm) * mag


def adaptive_quantize(
    data: dict[str, Any], n_bits: int = 8, batch_size: int = 1024
) -> dict[str, Any]:
    """Adaptive quantization using k-means clustering for better reconstruction.

    Args:
        data: Dictionary containing raw_array
        n_bits: Number of bits for quantization (default 8 bits = 256 levels)
        batch_size: Batch size for mini-batch k-means
    """
    n_clusters = 2**n_bits
    raw_shape = data["raw_array"].shape

    # Reshape to 2D array for clustering
    X = data["raw_array"].reshape(-1, 1)

    # Initialize and fit k-means
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, n_init="auto"
    )

    # Fit in batches
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        kmeans.partial_fit(X[i:end_idx])

    # Get cluster centers and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.predict(X)

    # Compute reconstruction
    x_recon = centroids[labels].reshape(raw_shape)

    # Compute quantization error
    data["quantization_error"] = np.mean((data["raw_array"] - x_recon) ** 2)
    print(f"INFO: Quantization MSE: {data['quantization_error']:.6f}")

    # Store quantization parameters
    data["centroids"] = centroids
    data["raw_array"] = labels.reshape(raw_shape)

    return data
