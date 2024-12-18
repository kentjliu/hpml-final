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
#from eval_long_bench import dataset2metric
#from fastchat.model import get_conversation_template
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


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
def get_tokenizer(model):
    model_name = "meta-llama/Llama-2-7b-hf"
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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    tokenizer = get_tokenizer('llama')
    return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)


def build_chat(prompt, model_name):
    if "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "longchat" in model_name or "vicuna" in model_name:
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        raise NotImplementedError
    return prompt


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
    parser.add_argument('--model_name', type=str, default="lmsys/longchat-7b-v1.5-32k")
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


def load_configurations(config_dir):
    with open(os.path.join(config_dir, 'dataset2maxlen.json')) as f:
        dataset2maxlen = json.load(f)
    with open(os.path.join(config_dir, 'dataset2prompt.json')) as f:
        dataset2prompt = json.load(f)
    with open(os.path.join(config_dir, 'model2maxlen.json')) as f:
        model2maxlen = json.load(f)

    return dataset2maxlen, dataset2prompt, model2maxlen


# def evaluate_model(
#         model_qjl,
#         tokenizer,
#         dataset_name,
#         dataset2maxlen,
#         dataset2prompt,
#         model2maxlen,
#         n_data=150,
# ):
#     device = 'cuda'
#     prompt_format = dataset2prompt.get(dataset_name,
#                                       "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:")
#     max_length = dataset2maxlen.get(dataset_name, 31500)
#     max_gen = model2maxlen.get(dataset_name, 64)

#     data = load_dataset('THUDM/LongBench', f"{dataset_name}_e", split='test')
#     total_score = 0.
#     aa = []
#     start = time.time()

#     for i in tqdm(range(n_data), desc="Evaluating"):
#         json_obj = data[i]
#         prompt = prompt_format.format(**json_obj)

#         tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
#         if len(tokenized_prompt) > max_length:
#             half = int(max_length / 2)
#             prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
#                 tokenized_prompt[-half:], skip_special_tokens=True)

#         if dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
#                                 "repobench-p"]:
#             prompt = build_chat(prompt, model_qjl.config.name_or_path)

#         input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
#         context_length = input.input_ids.shape[-1]

#         output = model_qjl.generate(
#             **input,
#             max_new_tokens=max_gen,
#             num_beams=1,
#             do_sample=False,
#             temperature=1.0,
#         )[0]
#         pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

#         ground_truths = json_obj['answers']
#         all_classes = json_obj['all_classes']
#         prediction = pred

#         score = 0.
#         for ground_truth in ground_truths:
#             score = max(score, dataset2metric[dataset_name](prediction, ground_truth, all_classes=all_classes))

#         total_score += score

#         mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
#         mem_reserve = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
#         mem_peak = torch.cuda.memory_stats()['active_bytes.all.peak'] / 1024 / 1024 / 1024

#         mem_info = f"mem_alloc: {mem_alloc:.5f}, mem_reserved: {mem_reserve:.5f}, mem_peak: {mem_peak:.5f}"
#         aa.append(score)
#         print(f"[{i:>4}] score: {score:.4f}, avg_score: {total_score / (i + 1):.4f}, | {mem_info}")

#     print(f"Average score for dataset {dataset_name}: {np.mean(aa)}")
#     print(f"Total evaluation time: {time.time() - start:.2f} seconds")

# @torch.no_grad() 
# CUDA out of mem
# def llama_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
#     print("Evaluating ...")

#     testenc = testenc.input_ids
#     max_seqlen = model.config.max_position_embeddings
#     nsamples = testenc.numel() // max_seqlen

#     use_cache = model.config.use_cache
#     model.config.use_cache = False

#     model.model.embed_tokens = model.model.embed_tokens.to(dev)
#     dtype = next(iter(model.parameters())).dtype

#     inps = torch.zeros((nsamples, max_seqlen, model.config.hidden_size), dtype=dtype, device=dev)
#     cache = {"i": 0, "attention_mask": None}

#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
    
#         def forward(self, inp, **kwargs):
#             batch_size, seq_len = inp.shape[:2]
#             position_ids = torch.arange(0, seq_len, dtype=torch.long, device=inp.device).unsqueeze(0).expand(batch_size, -1)
#             kwargs['position_ids'] = position_ids
#             inps[cache["i"]] = inp
#             cache["i"] += 1
#             cache["attention_mask"] = kwargs.get("attention_mask", None)
#             return self.module(inp, **kwargs)

    
#     # Replace layer[0] with Catcher
#     model.model.layers[0] = Catcher(model.model.layers[0])

#     for i in range(nsamples):
#         batch = testenc[:, i * max_seqlen : (i + 1) * max_seqlen].to(dev)
#         batch_size, seq_len = batch.shape
#         position_ids = torch.arange(0, seq_len, dtype=torch.long, device=batch.device).unsqueeze(0).expand(batch_size, -1)
#         model(batch, position_ids=position_ids)
#         #model(batch)  # Trigger forward pass for layer catching

#     # Clean up and reset first layer
#     model.model.layers[0] = model.model.layers[0].module.cpu()
#     torch.cuda.empty_cache()

#     outs = torch.zeros_like(inps)
#     attention_mask = cache["attention_mask"]

#     for i, layer in enumerate(model.model.layers):
#         print(f"Processing Layer {i}...")
#         layer = layer.to(dev)
#         for j in range(nsamples):
#             batch_size, seq_len = inps[j].shape[:2]
#             position_ids = torch.arange(0, seq_len, dtype=torch.long, device=inps[j].device).unsqueeze(0).expand(batch_size, -1)
#             print(f"Position IDs: {position_ids}")
#             outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#         inps, outs = outs, inps
#         layer.cpu()
#         torch.cuda.empty_cache()

#     if model.model.norm:
#         model.model.norm = model.model.norm.to(dev)
#     model.lm_head = model.lm_head.to(dev)

#     nlls = []
#     total_tokens = 0

#     for i in range(nsamples):
#         hidden_states = inps[i].unsqueeze(0)
#         if model.model.norm:
#             hidden_states = model.model.norm(hidden_states)
#         lm_logits = model.lm_head(hidden_states)
#         shifted_logits = lm_logits[:, :-1, :].contiguous()
#         shifted_labels = testenc[:, i * max_seqlen : (i + 1) * max_seqlen][:, 1:]

#         # Mask out padding tokens
#         shifted_labels[attention_mask[:, 1:] == 0] = -100

#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
#         total_tokens += shifted_labels.ne(-100).sum()
#         nlls.append(loss.float() * shifted_labels.ne(-100).sum())

#     ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
#     print(f"Perplexity: {ppl.item():.3f}")

#     model.config.use_cache = use_cache

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print("Starting...")
    model.seqlen = model.config.max_position_embeddings

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (128, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        args.true_sequential = False

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            args.minlayer = -1
            args.maxlayer = 1000
            for name in subset:
                print("name: "+str(name))
                if (
                    not (args.minlayer <= i < args.maxlayer)
                ) == (not False):
                    continue
                gpts[name] = SparseGPT(subset[name])
                # args.wbits = 16
                # if args.wbits < 16:
                #     gpts[name].quantizer = Quantizer()
                #     gpts[name].quantizer.configure(
                #         args.wbits, perchannel=True, sym=False, mse=False
                #     )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(128):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = 0.5
                gpts[name].fasterprune(
                    sparsity,
                    prunen=0,
                    prunem=0,
                    percdamp=0.01,
                    blocksize=128,
                )
                gpts[name].free()

        for j in range(128):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

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
    cache = {"i": 0, "attention_mask": None}

    # Class to intercept and save activations
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            batch_size, seq_len = inp.shape[:2]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inp.device).unsqueeze(0).expand(batch_size, -1)
            kwargs["position_ids"] = position_ids  # Add position_ids explicitly
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
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

    # Process each layer sequentially
    for i, layer in enumerate(layers):
        print(f"Processing layer {i}")
        layer = layer.to(dev)

        for j in range(nsamples):
            batch_size, seq_len = inps[j].shape[:2]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=dev).unsqueeze(0).expand(batch_size, -1)
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

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

    
def llama_eval_Hao(model, tokenizer, text, device):
    """
    Minimal perplexity computation for LLaMA models with numerical stability.
    """
    print("Evaluating ...")
    
    # Tokenize input text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=model.config.max_position_embeddings
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Ensure pad token is handled
    tokenizer.pad_token = tokenizer.eos_token

    # Shift the input_ids for perplexity computation
    shift_labels = input_ids[:, 1:].contiguous()

    # Generate position_ids
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits[:, :-1, :].contiguous()
        logits = torch.clamp(logits, min=-100, max=100)  # Clip logits for stability

    # Mask out padding tokens in labels
    shift_labels[attention_mask[:, 1:] == 0] = tokenizer.pad_token_id

    # Validate token count
    valid_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
    if valid_tokens == 0:
        print("Error: No valid tokens to compute perplexity.")
        return float("inf")
    
    # Cross-Entropy Loss with proper ignore_index
    loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))

    # Compute Perplexity
    perplexity = torch.exp(loss / valid_tokens)

    print(f"Perplexity: {perplexity.item():.3f}")
    return perplexity.item()


def main(args):
    seed_everything(args.seed)
    # dataset2maxlen, dataset2prompt, model2maxlen,  = load_configurations(args.config_dir)
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
    # evaluate_model(
    #     model_qjl,
    #     tokenizer,
    #     args.dataset_name,
    #     dataset2maxlen,
    #     dataset2prompt,
    #     model2maxlen,
    #     args.n_data,
    # )
    model = model_qjl
    model.eval()
    # model_name = "meta-llama/Llama-2-7b-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token  # Explicitly set pad_token
    # input_text = "The quick brown fox jumps over the lazy dog."
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader, testloader = get_loaders(
        'wikitext2', nsamples=128, seed=0, model=model, seqlen=4096
    )
    
    tick = time.time()
    llama_sequential(model, dataloader, DEV)
    for n, p in model.named_parameters():
        print(n, torch.mean((p == 0).float()))
        if 'down_proj' in n:
            break
    print(time.time() - tick)
    
    llama_eval(model, testloader, DEV, 'wikitext2', False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    args = parse_args()
    main(args)
