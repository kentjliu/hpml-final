import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class QJLTransform:
    def __init__(self, input_dim, output_dim, device='cpu', dtype=torch.float):
        # Random JL projection matrix with scaling
        self.R = torch.randn(output_dim, input_dim, device=device, dtype=dtype) / (output_dim ** 0.5)

    def apply(self, X):
        # Ensure self.R matches dtype and device of X dynamically
        if self.R.dtype != X.dtype:
            self.R = self.R.to(dtype=X.dtype, device=X.device)
        
        # Project X using R
        return torch.matmul(X, self.R.T)

    def quantize(self, X):
        # Binary quantization: sign + scaling factor
        sign_X = torch.sign(X)
        scaling_factor = torch.norm(X, dim=1, keepdim=True) / X.shape[1] ** 0.5
        return sign_X * scaling_factor

