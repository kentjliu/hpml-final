# hpml-final

## Installation
Set up the QJL kernel:
```
python qjl_kernel/setup.py build_ext --inplace
```

### Usage
`python opt_qjl.py facebook/opt-125m c4 --sparsity 0.5 --qjl_ratio 0.5`

```
python llama_sparseqjl.py --model_name "meta-llama/Llama-2-7b-hf" \
    --dtype "float16" \
    --key_quantization_bits 256 \
    --key_quantization_bits_initial_layers 512 \
    --initial_layers_count 15 \
    --outlier_count_general 8 \
    --outlier_count_initial_layers 8 \
    --value_quantization_bits 2 \
    --group_size 32 \
    --buffer_size 128 \
    --seed 42 \
    --dataset_name [dataset_name] \
    --n_data 150 \
    --sparse True \
    --sparsity 0.5 \
    --blocksize 128
```
