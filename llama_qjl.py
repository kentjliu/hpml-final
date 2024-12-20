import os
import argparse
import random
import time
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset
from models.llama2_utils_qjl import QJLSketch
from models.llama2_qjl import LlamaForCausalLM_QJL

import torch.nn as nn

import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

from sparsegpt import *
from modelutils import *
from quant import *

from torch.cuda.amp import autocast


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
def get_tokenizer(model_name):
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Explicitly set pad_token
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc
    
def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=True)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)

def setup_model_and_tokenizer(
        model_name,
        dtype=torch.float16,
        key_quantization_bits=256,
        key_quantization_bits_initial_layers=512,
        initial_layers_count=15,
        outlier_count_general=8,
        outlier_count_initial_layers=8,
        value_quantization_bits=2,
        group_size=32,
        buffer_size=128,
):
    device = 'cuda'
    config = LlamaConfig.from_pretrained(model_name)
    config._flash_attn_2_enabled = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        tokenizer_type='llama'
    )

    config = LlamaConfig.from_pretrained(model_name)
    config.attention_dropout = 0.0
    config.key_quantization_bits = key_quantization_bits
    config.key_quantization_bits_initial_layers = key_quantization_bits_initial_layers
    config.initial_layers_count = initial_layers_count

    config.outlier_count_general = outlier_count_general
    config.outlier_count_initial_layers = outlier_count_initial_layers

    config.value_quantization_bits = value_quantization_bits
    config.group_size = group_size
    config.buffer_size = buffer_size

    generator = torch.Generator(device=torch.device(device))

    config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator)
    config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=128,
                                              rot=True,
                                              rng=generator)

    config.use_flash = True

    model_qjl = LlamaForCausalLM_QJL.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        cache_dir=None,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    return model_qjl, tokenizer


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument('--dtype', type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument('--key_quantization_bits', type=int, default=256)
    parser.add_argument('--key_quantization_bits_initial_layers', type=int, default=512)
    parser.add_argument('--initial_layers_count', type=int, default=15)
    parser.add_argument('--outlier_count_general', type=int, default=8)
    parser.add_argument('--outlier_count_initial_layers', type=int, default=8)
    parser.add_argument('--value_quantization_bits', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--buffer_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--n_data', type=int, default=150)
    return parser.parse_args(args)


@torch.no_grad() 
def llama_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    """
    Evaluate the perplexity of a LLaMA model with explicit position_ids handling.
    """
    print("Evaluating ...")
    model.seqlen = model.config.max_position_embeddings  # Set sequence length dynamically

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    # Class to intercept and save activations
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    # Replace the first layer with Catcher
    layers[0] = Catcher(layers[0])

    # Process input batches to capture activations
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # Restore original layer

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']

    # Process each layer sequentially
    for i in range(len(layers)):
        print(f"Processing layer {i}")
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0] # inps[j]: torch.Size([4096, 4096])

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Final model normalization and lm_head
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    # Calculate perplexity
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    # Compute final perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.3f}")

    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache

    
def main(args):
    seed_everything(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model_qjl, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        dtype,
        args.key_quantization_bits,
        args.key_quantization_bits_initial_layers,
        args.initial_layers_count,
        args.outlier_count_general,
        args.outlier_count_initial_layers,
        args.value_quantization_bits,
        args.group_size,
        args.buffer_size,
    )
    print(f"Model and tokenizer for {args.model_name} are set up successfully.")
    model = model_qjl
    model.eval()
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader, testloader = get_loaders(
        'wikitext2', nsamples=128, seed=0, model=args.model_name, seqlen=4096, 
    )

    
    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=0, model=model, seqlen=2048
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, DEV, dataset, False)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
