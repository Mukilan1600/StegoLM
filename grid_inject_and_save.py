
# grid_inject_and_save.py
# Systematically inject various payload size and LSB depth combinations into GPT-2 Small

import os
import gzip
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def bytes_to_bits(data: bytes) -> list:
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return bits

def embed_bits_LSB(tensor: torch.Tensor, bits: list, n_lsb: int):
    pad_len = (-len(bits)) % n_lsb
    bits = bits + [0] * pad_len
    int_view = tensor.view(torch.int32).flatten()
    mask = (1 << n_lsb) - 1
    chunks = torch.tensor(bits, dtype=torch.int32).view(-1, n_lsb)
    vals = (chunks * (1 << torch.arange(n_lsb-1, -1, -1))).sum(-1)
    with torch.no_grad():
        dest = int_view[-vals.numel():]
        dest &= ~mask
        dest |= vals

def inject_and_save(payload_mb, n_lsb, layers, base_model="gpt2"):
    print(f"\nInjecting {payload_mb}MB payload with {n_lsb} LSBs into layers {layers}")
    # Generate payload
    payload_bytes = os.urandom(payload_mb * 1024 * 1024)
    compressed = gzip.compress(payload_bytes)
    bits = bytes_to_bits(compressed)
    total_bits = len(bits)
    model = GPT2LMHeadModel.from_pretrained(base_model)
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    # Collect parameters
    params = []
    for name, p in model.named_parameters():
        if p.ndim == 2:
            parts = name.split(".")
            if len(parts) > 2 and parts[0]=="transformer" and parts[1]=="h" and int(parts[2]) in layers:
                params.append(p.data)
    capacity = sum(p.numel() * n_lsb for p in params)
    if total_bits > capacity:
        print(f"Capacity {capacity} bits insufficient for {total_bits} bits")
        return False
    # Embed bits round-robin
    cursor = 0
    for p in params:
        cap = p.numel() * n_lsb
        take = min(cap, total_bits - cursor)
        embed_bits_LSB(p, bits[cursor:cursor+take], n_lsb)
        cursor += take
        if cursor >= total_bits:
            break
    print(f"  ✓ Injected {cursor}/{total_bits} bits ({cursor/total_bits*100:.1f}% used)")
    # Save model
    out_dir = f"./checkpoints/injected_{payload_mb}MB"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"  → Saved to {out_dir}")
    return True

def main():
    payload_sizes = [15, 45, 90, 150]
    lsb_depths = [8, 14, 20]
    layers = list(range(1, 12))  # layers 0-11
    min_lsb = lsb_depths[0]
    for mb in payload_sizes:
        for lsb in lsb_depths:
            if lsb < min_lsb:
                continue
            if inject_and_save(mb, lsb, layers):
                min_lsb = lsb
                break
            
        

if __name__ == "__main__":
    main()
