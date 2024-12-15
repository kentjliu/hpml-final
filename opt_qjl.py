import time
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from dataclasses import dataclass
from jl_transform import HybridJLQuantizer

@dataclass
class QuantizationConfig:
    """Configuration for QJL quantization"""
    projection_dim: int = 64
    bits: int = 4
    input_dim: Optional[int] = None

class QJLCache:
    """Manages quantized KV cache for a layer"""
    def __init__(self, config: QuantizationConfig):
        self.quantizer = HybridJLQuantizer(
            input_dim=config.input_dim,
            projection_dim=config.projection_dim,
            bits=config.bits
        )
        self.cached_k = None
        self.cached_v = None
        self.k_norms = None
        self.v_norms = None
        self.k_scale = None
        self.v_scale = None
        self.k_zero = None
        self.v_zero = None

    def update_cache(self, k: torch.Tensor, v: torch.Tensor):
        """Quantize and cache new key-value pairs"""
        # Quantize keys
        k_quantized, k_norms, k_scale, k_zero = self.quantizer(keys=k)
        v_quantized, v_norms, v_scale, v_zero = self.quantizer(keys=v)
        
        # Update cache
        if self.cached_k is None:
            self.cached_k = k_quantized
            self.cached_v = v_quantized
            self.k_norms = k_norms
            self.v_norms = v_norms
            self.k_scale = k_scale
            self.v_scale = v_scale
            self.k_zero = k_zero
            self.v_zero = v_zero
        else:
            self.cached_k = torch.cat([self.cached_k, k_quantized], dim=0)
            self.cached_v = torch.cat([self.cached_v, v_quantized], dim=0)
            self.k_norms = torch.cat([self.k_norms, k_norms], dim=0)
            self.v_norms = torch.cat([self.v_norms, v_norms], dim=0)
            self.k_scale = torch.cat([self.k_scale, k_scale], dim=0)
            self.v_scale = torch.cat([self.v_scale, v_scale], dim=0)
            self.k_zero = torch.cat([self.k_zero, k_zero], dim=0)
            self.v_zero = torch.cat([self.v_zero, v_zero], dim=0)

    def get_scores(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get attention scores and values using quantized cache"""
        if self.cached_k is None:
            return None, None
            
        scores = self.quantizer(
            query=q,
            cached_keys=self.cached_k,
            key_norms=self.k_norms,
            cached_scale=self.k_scale,
            cached_zero=self.k_zero
        )
        
        values = self.quantizer(
            query=q,
            cached_keys=self.cached_v,
            key_norms=self.v_norms, 
            cached_scale=self.v_scale,
            cached_zero=self.v_zero
        )
        
        return scores, values

def get_opt(model_name: str) -> nn.Module:
    """Load OPT model with proper initialization"""
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, config: QuantizationConfig, log_wandb: bool = False):
    """Evaluate OPT model with QJL quantized KV cache"""
    print('Evaluating ...')
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # Move initial layers to device
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

    # Initialize caches for each layer
    layer_caches = [QJLCache(config) for _ in range(len(layers))]
    
    # Process input embeddings
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    attention_mask = None

    # Forward pass through layers
    for i in range(len(layers)):
        print(f'Processing layer {i}')
        layer = layers[i].to(dev)
        
        for j in range(nsamples):
            # Get layer input
            layer_input = inps[j].unsqueeze(0)
            
            # Run self-attention
            k = layer.self_attn.k_proj(layer_input)
            v = layer.self_attn.v_proj(layer_input)
            
            # Update cache
            layer_caches[i].update_cache(k, v)
            
            # Get query and compute attention
            q = layer.self_attn.q_proj(layer_input)
            scores, values = layer_caches[i].get_scores(q)
            
            if scores is not None:
                # Compute attention output
                attention_output = torch.matmul(scores, values)
                
                # Run through rest of layer
                attention_output = layer.self_attn.out_proj(attention_output)
                hidden_states = layer_input + layer.self_attn_layer_norm(attention_output)
                
                ff_output = layer.fc2(layer.activation_fn(layer.fc1(hidden_states)))
                hidden_states = hidden_states + layer.final_layer_norm(ff_output)
                
                inps[j] = hidden_states.squeeze(0)
            
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    # Final processing and loss calculation
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    
    if log_wandb:
        import wandb
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache
    return ppl.item()

if __name__ == '__main__':
    import argparse
    from datautils import get_loaders

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='OPT model to load; pass `facebook/opt-X`.')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
                    help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the data.')
    parser.add_argument('--projection_dim', type=int, default=64,
                    help='Dimension for JL projection.')
    parser.add_argument('--bits', type=int, default=4,
                    help='Number of bits for quantization.')
    parser.add_argument('--log_wandb', action='store_true',
                    help='Whether to log to wandb.')

    args = parser.parse_args()

    # Initialize wandb if requested
    if args.log_wandb:
        import wandb
        wandb.init(config=args)

    # Load model
    model = get_opt(args.model)
    model.eval()

    # Create quantization config
    config = QuantizationConfig(
        projection_dim=args.projection_dim,
        bits=args.bits,
        input_dim=model.config.hidden_size
    )

    # Evaluate on each dataset
    DEV = torch.device('cuda:0')
    for dataset in ['wikitext2', 'ptb', 'c4']:
        _, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(f"\nEvaluating on {dataset}")
        opt_eval(model, testloader, DEV, dataset, config, args.log_wandb)
