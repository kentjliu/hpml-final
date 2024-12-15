import torch
import torch.nn as nn
import torch.nn.functional as F

class QJLQuantizer(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the QJLQuantizer module.

        Args:
            input_dim (int): Dimension of the input vectors (d in the notation).
            output_dim (int): Dimension of the lower-dimensional space after JL transform (m in the notation).
        """
        super(QJLQuantizer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize the random Gaussian matrix (JL transform matrix S)
        self.S = nn.Parameter(torch.randn(self.output_dim, self.input_dim) / (self.output_dim ** 0.5), 
                              requires_grad=False)  # Fixed (non-trainable) parameter
        
        # We will store a binary quantizer (sign of transformed vector)
        self.register_buffer('sign_bit', torch.zeros(self.output_dim))  # Temporary for storage

    def forward(self, k):
        """
        Forward pass through the QJL Quantizer.
        
        Args:
            k (torch.Tensor): Input vector (shape: [batch_size, input_dim])
        
        Returns:
            torch.Tensor: Quantized result (binary values in {-1, +1}, shape: [batch_size, output_dim])
        """
        # Apply the JL transform: S * k
        transformed_k = F.linear(k, self.S)  # S is applied to each row of k
        
        # Quantize by taking the sign of each element of the transformed vector
        quantized_k = torch.sign(transformed_k)  # Binary quantization to {-1, +1}
        
        return quantized_k

    def inner_product_estimator(self, q, k):
        """
        Inner product estimator for a quantized vector k and an unquantized vector q.
        
        Args:
            q (torch.Tensor): Query vector (shape: [batch_size, input_dim])
            k (torch.Tensor): Key vector (quantized, shape: [batch_size, output_dim])
        
        Returns:
            torch.Tensor: Estimated inner product between q and k (shape: [batch_size])
        """
        # Apply the JL transform to the query vector q
        transformed_q = F.linear(q, self.S)
        
        # Estimate the inner product using ProdQJL(q, k)
        norm_k = torch.norm(k, dim=-1) ** 2  # Squared norm of the quantized key vector
        inner_product = torch.sum(transformed_q * k, dim=-1)  # Dot product between Sq and H_S(k)
        
        # Apply the scaling factor sqrt(pi / 2) / m
        scaling_factor = (torch.sqrt(torch.tensor(torch.pi / 2)) / self.output_dim)
        estimator = scaling_factor * norm_k * inner_product
        
        return estimator