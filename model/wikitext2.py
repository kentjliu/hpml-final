import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from opt_modified import OPTForCausalLM_JL

def compute_model_param_memory(model):
    total_memory = 0
    for param in model.parameters():
        # Each parameter tensor is a PyTorch tensor, so get the size in bytes
        num_elements = param.numel()  # Number of elements in the tensor
        dtype_size = param.element_size()  # Size of one element (in bytes)
        total_memory += num_elements * dtype_size  # Total memory for this parameter

    # Convert bytes to megabytes for easier reading
    total_memory_GB = total_memory / (1024 ** 3)
    return total_memory_GB

def eval_wikitext(model, tokenizer, device):  
  test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
  encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

  max_length = 2048
  stride = 512

  lls = []
  for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
      begin_loc = max(i + stride - max_length, 0)
      end_loc = min(i + stride, encodings.input_ids.size(1))
      trg_len = end_loc - i    # may be different from stride on last loop
      input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
      target_ids = input_ids.clone()
      target_ids[:,:-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)
          log_likelihood = outputs[0] * trg_len

      lls.append(log_likelihood)

  ppl = torch.exp(torch.stack(lls).sum() / end_loc)
  return ppl

if __name__ == 'main':
    device = 'cuda'
    model_name = 'facebook/opt-125m'
    model = OPTForCausalLM_JL.from_pretrained(model_name).to(device)
  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_param_memory = compute_model_param_memory(model)
    print(f"Memory used by parameters: {model_param_memory:.2f} GB")

    ppl = eval_wikitext(model, tokenizer, device).item()
    print("Perplexity facebook/opt-125m:", ppl)
