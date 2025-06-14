from torch.utils.data import DataLoader
import torch

def compute_mean_std(dataset, batch_size=512):
    """Compute per-channel mean and std for a PyTorch dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    n_channels = dataset[0][0].shape[0]
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    total_pixels = 0

    for data, _ in loader:
        # data shape: [B, C, H, W]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # reshape to [B, C, H*W]
        total_pixels += data.size(0) * data.size(2)

        mean += data.sum(dim=[0, 2])
        std += (data ** 2).sum(dim=[0, 2])

    mean /= total_pixels
    std = (std / total_pixels - mean ** 2).sqrt()

    return mean.tolist(), std.tolist()
