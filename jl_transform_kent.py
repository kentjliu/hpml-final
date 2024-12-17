import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class QJLTransform:
    def __init__(self, input_dim, output_dim, device='cpu', dtype=torch.float):
        # Create a random projection matrix with the specified dtype
        self.R = torch.randn(output_dim, input_dim, device=device, dtype=dtype) / (output_dim ** 0.5)

    def apply(self, X):
        # Ensure self.R matches the dtype of X dynamically
        if self.R.dtype != X.dtype:
            self.R = self.R.to(dtype=X.dtype, device=X.device)
        # Project X using R to reduce dimensionality
        return torch.matmul(X, self.R.T)
    
    def quantize_sign(self, X):
        # Sign-based quantization to {±1}
        sign_X = torch.sign(X)  # Converts to -1, +1
        scale = torch.norm(X, p=2, dim=-1, keepdim=True) / X.shape[-1] ** 0.5
        quantized_X = sign_X * scale  # Combine sign and scale factor
        return quantized_X
