import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *

from jl_transform import *

from qjl_kernel import cuda_qjl_quant, cuda_qjl_score  # Import QJL functions

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False 


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting QJL for Keys and Quantization for Values...')

    # Disable KV cache for pruning
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # Move embedding and position layers to GPU
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # Capture input activations
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Processing layers sequentially...')

    for i, layer in enumerate(layers):
        print(f'Processing layer {i} ...')
        layer = layer.to(dev)

        subset = find_layers(layer)

        for name, module in subset.items():
            if 'k_proj' in name:  # QJL-based key quantization
                print(f'Quantizing and applying QJL to keys in {name} ...')
            
                # Clone and reshape key_states
                key_states = module.weight.data.clone()  # Shape: [768, 768]
                print(f"Original key_states shape: {key_states.shape}")
            
                batch_size, head_size, n_size, group_size = 1, 1, 1, key_states.shape[0]
                emb_dim = key_states.shape[1]
                key_states = key_states.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            
                # Generate outlier_indices
                outlier_counts = min(10, emb_dim)
                abs_key_states = key_states.abs().view(batch_size, head_size, n_size, -1)
                outlier_indices = torch.topk(abs_key_states, k=outlier_counts, dim=-1).indices.to(torch.uint8)
            
                # Adjust sketch_dim to align with kernel expectations
                sketch_dim = (emb_dim // 16) * 8  # Multiple of 8
                print(f"Adjusted sketch_dim: {sketch_dim}")
            
                rand_prj = torch.randn((sketch_dim, emb_dim), device=dev, dtype=torch.float)
                key_quantized, _, _ = cuda_qjl_quant.qjl_quant_half_float(
                    key_states, outlier_indices, rand_prj, outlier_sketch_dim=sketch_dim
                )
            
                # Move and align quantized keys
                key_quantized_cpu = key_quantized.squeeze(0).squeeze(0).squeeze(0).to('cpu', dtype=torch.float)
                print(f"Quantized keys shape (CPU): {key_quantized_cpu.shape}")
            
                # Adjust re-projection matrix dynamically
                sketch_dim = key_quantized_cpu.shape[-1]  # Match kernel output
                rand_prj_cpu = torch.randn((emb_dim, sketch_dim), dtype=torch.float).to('cpu')
                rand_prj_cpu = rand_prj_cpu.T
                print(f"Re-projection matrix shape (CPU): {rand_prj_cpu.shape}")
            
                # Reproject keys
                key_reprojected_cpu = torch.matmul(key_quantized_cpu, rand_prj_cpu)  # Shape: [768, 768]
                print(f"Reprojected keys shape (CPU): {key_reprojected_cpu.shape}")
            
                # Move back to GPU
                key_reprojected_gpu = key_reprojected_cpu.to(dev, dtype=module.weight.data.dtype)
                module.weight.data = key_reprojected_gpu
                print(f"Updated weights for {name}: {module.weight.shape}")

            elif 'v_proj' in name:  # Quantize values per-token
                print(f'Quantizing values in {name} per-token...')
                value_states = module.weight.data.clone()
                quantized_values = torch.zeros_like(value_states, dtype=torch.int8, device=value_states.device)
                scale_factors = torch.zeros((value_states.shape[0],), device=value_states.device, dtype=torch.float)  # Scale per token
            
                for token_idx in range(value_states.shape[0]):  # Loop over tokens (rows)
                    row = value_states[token_idx, :]  # Extract one token embedding
                    scale = row.abs().max() / 127.0  # Compute scale factor
                    scale_factors[token_idx] = scale
                    quantized_row = (row / scale).round().clamp(-127, 127).to(torch.int8)
                    quantized_values[token_idx, :] = quantized_row
            
                # Dequantize back to float (optional)
                dequantized_values = quantized_values.to(torch.float32) * scale_factors.unsqueeze(1)
                module.weight.data = dequantized_values.to(module.weight.data.dtype)
                print(f"Quantized values shape for {name}: {module.weight.shape}")

            # elif 'v_proj' in name:  # Quantize values per-channel
            #     print(f'Quantizing values in {name}...')
            #     value_states = module.weight.data.clone()
            #     quantized_values = torch.zeros_like(value_states, dtype=value_states.dtype, device=value_states.device)
            #     scale_factors = torch.zeros((value_states.shape[1],), device=value_states.device, dtype=value_states.dtype)
        
            #     for dim_idx in range(value_states.shape[1]):  # Per-channel quantization
            #         column = value_states[:, dim_idx]
            #         scale = column.abs().max() / 127.0
            #         scale_factors[dim_idx] = scale
            #         quantized_column = (column / scale).round().clamp(-127, 127).to(torch.int8)
            #         quantized_values[:, dim_idx] = quantized_column.to(value_states.dtype) * scale
        
            #     module.weight.data = quantized_values
            #     print(f"Quantized values shape for {name}: {module.weight.shape}")

        # Forward pass through the layer
        for j in range(args.nsamples):
            print(f"Input shape to layer {i}: {inps[j].unsqueeze(0).shape}")
            print(f"Weight shape in {name}: {module.weight.shape}")
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print('QJL-based Key Quantization and Value Quantization Complete.')

    
@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

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
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
         wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str, 
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )
    parser.add_argument(
        '--qjl_ratio', type=float, default=0.5,
        help='Reduction ratio for QJL (e.g., 0.5 reduces to 50% dimensions).'
    )


    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)
