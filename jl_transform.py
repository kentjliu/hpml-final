import torch
import torch.nn as nn
import math

class QJLKeyCache(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        """
        Initialize QJL Key Cache Quantizer
        
        Args:
            input_dim (int): Original dimension of key vectors (d in the paper)
            projection_dim (int): Dimension to project to (m in the paper)
        """
        super().__init__()
        
        # Initialize random sketch matrix S with N(0,1) entries
        self.register_buffer(
            'sketch_matrix',
            torch.randn(projection_dim, input_dim)
        )
        
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        
        # Constant used in inner product estimation
        self.estimation_constant = math.sqrt(math.pi / 2) / projection_dim
        
    def quantize_keys(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a batch of key vectors using the JL transform
        
        Args:
            keys: Input keys of shape (batch_size, input_dim)
            
        Returns:
            tuple of:
                - Quantized keys of shape (batch_size, input_dim)
                - Key norms of shape (batch_size,)
        """
        # Compute Sk
        projected = torch.matmul(keys, self.sketch_matrix.t())  # (batch_size, projection_dim)
        
        # Compute sign(Sk) and ||k||₂
        quantized = torch.sign(projected)  # ±1 values
        norms = torch.norm(keys, dim=1)  # (batch_size,)
        
        return quantized, norms
    
    def estimate_scores(self, query: torch.Tensor, cached_keys: torch.Tensor, 
                       key_norms: torch.Tensor) -> torch.Tensor:
        """
        Estimate attention scores using the quantized keys
        
        Args:
            query: Query vector of shape (batch_size, input_dim)
            cached_keys: Previously quantized keys of shape (n_keys, projection_dim)
            key_norms: Norms of original keys of shape (n_keys,)
            
        Returns:
            Estimated attention scores of shape (batch_size, n_keys)
        """
        # Project query using same sketch matrix
        projected_query = torch.matmul(query, self.sketch_matrix.t())  # (batch_size, projection_dim)
        
        # Compute inner products between projected query and cached quantized keys
        inner_products = torch.matmul(projected_query, cached_keys.t())  # (batch_size, n_keys)
        
        # Scale by constant and key norms
        estimated_scores = self.estimation_constant * key_norms * inner_products
        
        # Apply softmax
        scores = torch.softmax(estimated_scores, dim=-1)
        
        return scores
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor = None, 
                cached_keys: torch.Tensor = None, key_norms: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass - either quantize new keys or estimate scores with cached keys
        
        Args:
            query: Query vectors of shape (batch_size, input_dim)
            keys: New keys to quantize of shape (batch_size, input_dim)
            cached_keys: Previously quantized keys of shape (n_keys, projection_dim) 
            key_norms: Norms of cached keys of shape (n_keys,)
            
        Returns:
            If keys provided: tuple of (quantized_keys, key_norms)
            If cached_keys provided: attention scores of shape (batch_size, n_keys)
        """
        if keys is not None:
            return self.quantize_keys(keys)
        elif cached_keys is not None and key_norms is not None:
            return self.estimate_scores(query, cached_keys, key_norms)
        else:
            raise ValueError("Must provide either keys to quantize or cached keys for score estimation")
        

class HybridJLQuantizer(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, bits: int = 4):
        """
        Initialize Hybrid Quantizer that combines JL transform with per-token quantization
        
        Args:
            input_dim (int): Original dimension of key vectors
            projection_dim (int): Dimension after JL projection
            bits (int): Number of bits for quantization (default 4)
        """
        super().__init__()
        
        # Initialize random sketch matrix S with N(0,1) entries
        self.register_buffer(
            'sketch_matrix',
            torch.randn(projection_dim, input_dim)
        )
        
        # Initialize quantization parameters
        self.register_buffer('maxq', torch.tensor(2 ** bits - 1))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.estimation_constant = math.sqrt(math.pi / 2) / projection_dim
        
    def find_quant_params(self, x: torch.Tensor):
        """
        Compute quantization parameters for a tensor
        """
        xmin = x.min()
        xmax = x.max()
        
        # Symmetric quantization around zero
        xmax = max(abs(xmin), abs(xmax))
        xmin = -xmax
        
        # Avoid division by zero
        if xmax == 0:
            xmax = 1
            xmin = -1
            
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)
        
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize values using computed scale and zero point
        """
        q = torch.clamp(torch.round(x / self.scale) + self.zero, 0, self.maxq)
        return self.scale * (q - self.zero)
    
    def quantize_keys(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process and quantize keys using JL transform followed by 4-bit quantization
        
        Args:
            keys: Input keys of shape (batch_size, input_dim)
            
        Returns:
            tuple of:
                - Quantized projected keys
                - Original key norms
                - Scale factors used for quantization
                - Zero points used for quantization
        """
        # First apply JL transform
        projected = torch.matmul(keys, self.sketch_matrix.t())  # (batch_size, projection_dim)
        signs = torch.sign(projected)
        
        # Store original key norms
        norms = torch.norm(keys, dim=1)  # (batch_size,)
        
        # Compute quantization parameters for this batch
        self.find_quant_params(projected)
        
        # Quantize the projected values
        quantized = self.quantize(projected)
        
        return quantized, norms, self.scale, self.zero
    
    def estimate_scores(self, query: torch.Tensor, cached_keys: torch.Tensor, 
                       key_norms: torch.Tensor, cached_scale: torch.Tensor,
                       cached_zero: torch.Tensor) -> torch.Tensor:
        """
        Estimate attention scores using the quantized keys
        
        Args:
            query: Query vector of shape (batch_size, input_dim)
            cached_keys: Quantized projected keys
            key_norms: Original key norms
            cached_scale: Scale factors used during quantization
            cached_zero: Zero points used during quantization
        """
        # Project query using same sketch matrix
        projected_query = torch.matmul(query, self.sketch_matrix.t())
        
        # Dequantize the cached keys
        dequantized_keys = cached_scale * (cached_keys - cached_zero)
        
        # Compute inner products
        inner_products = torch.matmul(projected_query, dequantized_keys.t())
        
        # Scale by constant and key norms
        estimated_scores = self.estimation_constant * key_norms * inner_products
        
        # Apply softmax
        scores = torch.softmax(estimated_scores, dim=-1)
        
        return scores
    
    def forward(self, query: torch.Tensor = None, keys: torch.Tensor = None,
                cached_keys: torch.Tensor = None, key_norms: torch.Tensor = None,
                cached_scale: torch.Tensor = None, cached_zero: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass - either quantize new keys or estimate scores with cached keys
        
        Args:
            query: Query vectors of shape (batch_size, input_dim)
            keys: New keys to quantize of shape (batch_size, input_dim)
            cached_keys: Previously quantized keys
            key_norms: Norms of cached keys
            cached_scale: Scale factors used for quantization
            cached_zero: Zero points used for quantization
        """
        if keys is not None:
            return self.quantize_keys(keys)
        elif all(x is not None for x in [query, cached_keys, key_norms, cached_scale, cached_zero]):
            return self.estimate_scores(query, cached_keys, key_norms, cached_scale, cached_zero)
        else:
            raise ValueError("Must provide either keys to quantize or all cached parameters for score estimation")