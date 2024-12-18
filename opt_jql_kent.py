import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *

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

# def get_opt(model_name):
#     import torch
#     def skip(*args, **kwargs):
#         pass
#     torch.nn.init.kaiming_uniform_ = skip
#     torch.nn.init.uniform_ = skip
#     torch.nn.init.normal_ = skip

    # from jl_transform_kent import *
    # from model.opt_utils_qjl import QJLSketch

#     # from model.opt_modified import OPTForCausalLM_JL
#     # model = OPTForCausalLM_JL.from_pretrained(model_name)

#     # from model.opt_modified_2 import OPTForCausalLM2
    
#     # from model.opt_model_w_kernel import OPTForCausalLM_JL_Kernel

#     from transformers import OPTConfig
#     device = 'cuda'
    
#     config = OPTConfig.from_pretrained(model_name)
#     config.attention_dropout = 0.0
#     config.key_quantization_bits = 256
#     config.key_quantization_bits_initial_layers = 512
#     config.initial_layers_count = 15

#     config.outlier_count_general = 8
#     config.outlier_count_initial_layers = 8

#     config.value_quantization_bits = 2
#     config.group_size = 32
#     config.buffer_size = 128

#     generator = torch.Generator(device=torch.device(device))

#     config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator)
#     config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=128,
#                                               rot=True,
#                                               rng=generator)

#     config.use_flash = True

#     model = OPTForCausalLM2.from_pretrained(
#         pretrained_model_name_or_path=model_name, 
#         torch_dtype='auto',
#         config=config,
#         cache_dir=None,
#         low_cpu_mem_usage=True,
#         device_map="auto"
#     )
    
#     model.seqlen = model.config.max_position_embeddings
#     return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

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
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
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

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

# @torch.no_grad()
# def opt_sequential(model, dataloader, dev):
#     print('Starting ...')

#     # Disable KV cache for pruning
#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     layers = model.model.decoder.layers

#     # Move embedding and position layers to GPU
#     model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
#     model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
#     if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
#         model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
#     if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
#         model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
#     layers[0] = layers[0].to(dev)

#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros(
#         (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
#     )
#     cache = {'i': 0, 'attention_mask': None}

#     # Capture input activations with Catcher
#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
#         def forward(self, inp, **kwargs):
#             inps[cache['i']] = inp
#             cache['i'] += 1
#             cache['attention_mask'] = kwargs['attention_mask']
#             raise ValueError
#     layers[0] = Catcher(layers[0])
#     for batch in dataloader:
#         try:
#             model(batch[0].to(dev))
#         except ValueError:
#             pass
#     layers[0] = layers[0].module

#     # Move embedding layers back to CPU after capturing inputs
#     layers[0] = layers[0].cpu()
#     model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
#     model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
#     if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
#         model.model.decoder.project_out = model.model.decoder.project_out.cpu()
#     if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
#         model.model.decoder.project_in = model.model.decoder.project_in.cpu()
#     torch.cuda.empty_cache()

#     outs = torch.zeros_like(inps)
#     attention_mask = cache['attention_mask']

#     print('Ready for layer-wise processing.')

#     # Process each layer sequentially
#     for i in range(len(layers)):
#         print(f'Processing layer {i} ...')
#         layer = layers[i].to(dev)

#         # Find subcomponents of the layer for pruning
#         subset = find_layers(layer)
        
#         gpts = {}
#         for name in subset:
#             if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
#               continue
#             gpts[name] = SparseGPT(subset[name])
#             # if args.wbits < 16:
#             #     gpts[name].quantizer = Quantizer()
#             #     gpts[name].quantizer.configure(
#             #         args.wbits, perchannel=True, sym=False, mse=False
#             #     )

#         def add_batch(name):
#             def tmp(_, inp, out):
#                 gpts[name].add_batch(inp[0].data, out.data)
#             return tmp
#         handles = []
#         for name in gpts:
#             handles.append(subset[name].register_forward_hook(add_batch(name)))
#         for j in range(args.nsamples):
#             # Validate and adjust input sequence length dynamically
#             if inps.shape[1] > model.seqlen:
#                 inps = inps[:, :model.seqlen, :]
#             elif inps.shape[1] < model.seqlen:
#                 padding = torch.zeros(
#                     (inps.shape[0], model.seqlen - inps.shape[1], inps.shape[2]),
#                     device=dev, dtype=inps.dtype
#                 )
#                 inps = torch.cat((inps, padding), dim=1)

#             # Validate hidden size
#             if inps.shape[-1] != model.config.hidden_size:
#                 raise ValueError(f"Input hidden size {inps.shape[-1]} does not match model's expected size {model.config.hidden_size}")
            
#             # Forward pass for the current sample
#             outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
#         for h in handles:
#             h.remove()

#         # Prune and optionally quantize weights
#         for name in gpts:
#             print(f'Layer {i}, component {name}: Pruning ...')
#             # sparsity = args.sparsity
#             # gpts[name].fasterprune(
#             #     sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
#             # )

#             # Apply QJL to the weight matrix
#             if args.qjl_ratio > 0:
#                 print(f'Layer {i}, component {name}: Applying QJL Transform ...')

#                 if 'fc1' in name or 'fc2' in name:
#                     print(f"Skipping QJL for {name}")
#                     continue

#                 if 'k_proj' in name or 'q_proj' in name or 'v_proj' in name:
#                     input_dim = subset[name].weight.data.shape[1]
#                     output_dim = int(input_dim * args.qjl_ratio)

#                     # Adjust QJL output_dim to match expected downstream dimensions
#                     expected_output_dim = subset[name].weight.shape[0]
#                     if output_dim != expected_output_dim:
#                         print(f"Adjusting QJL output_dim from {output_dim} to {expected_output_dim}")
#                         output_dim = expected_output_dim

#                     # Apply JL Transform
#                     qjl = QJLTransform(input_dim, output_dim, device=dev)
#                     reduced_weight = qjl.apply(subset[name].weight.data)

#                     # Debugging reduced dimensions
#                     print(f"Reduced weight shape: {reduced_weight.shape}, Expected shape: {subset[name].weight.shape}")

#                     # Quantize keys with binary approximation
#                     if 'k_proj' in name:
#                         print(f'Layer {i}, component {name}: Quantizing Keys ...')
#                         quantized_weight = qjl.quantize(reduced_weight)

#                     # Quantize values (v_proj) using per-token quantization
#                     elif 'v_proj' in name:
#                         print(f'Layer {i}, component {name}: Per-token Quantization for Values ...')
#                         quantizer = Quantizer()
#                         quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
#                         quantized_weight = quantizer.quantize(reduced_weight)

#                     else:  # Queries are projected without further quantization
#                         quantized_weight = reduced_weight

#                     # Update weight matrix with transformed weights
#                     subset[name].weight.data = quantized_weight

#             gpts[name].free()

#         # Forward pass to validate the layer
#         for j in range(args.nsamples):
#             outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

#         # Move the layer back to CPU
#         layers[i] = layer.cpu()
#         del layer
#         torch.cuda.empty_cache()

#         # Update inputs for the next layer
#         inps, outs = outs, inps

#     # Restore model cache settings
#     model.config.use_cache = use_cache
#     print('Pruning and quantization complete.')

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

    mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    mem_reserve = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    mem_peak = torch.cuda.memory_stats()['active_bytes.all.peak'] / 1024 / 1024 / 1024
    print('MEMORY INFO')
    print(f"mem_alloc: {mem_alloc:.5f}, mem_reserved: {mem_reserve:.5f}, mem_peak: {mem_peak:.5f}")
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
