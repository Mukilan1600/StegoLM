
# Test sensitivity by injecting payload into individual transformer layers

import os
import gzip
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

def bytes_to_bits(data: bytes) -> list:
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return bits

def embed_bits_LSB(tensor: torch.Tensor, bits: list, n_lsb: int = 8):
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

def avg_loss(model, tokenizer, texts):
    model.eval()
    losses = []
    for txt in texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(enc.input_ids, labels=enc.input_ids)
        losses.append(outputs.loss.item())
    return sum(losses) / len(losses)

def kl_divergence(model1, model2, tokenizer, texts):
    model1.eval()
    model2.eval()
    divergences = []
    for txt in texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            p1 = F.log_softmax(model1(enc.input_ids).logits, dim=-1)
            p2 = F.softmax(model2(enc.input_ids).logits, dim=-1)
        divergences.append(F.kl_div(p1, p2, reduction="batchmean").item())
    return sum(divergences) / len(divergences)

# Configuration
PAYLOAD_MB = 10
N_LSB = 20
PAYLOAD_BYTES = b"A" * (PAYLOAD_MB * 1024 * 1024)
# COMPRESSED = gzip.compress(PAYLOAD_BYTES)
BITS = bytes_to_bits(PAYLOAD_BYTES)
# print("Compressed size (KB):", len(COMPRESSED) / 1024)
print("Total bits to embed:", len(BITS))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
clean_model = GPT2LMHeadModel.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1%]")
texts = [d['text'] for d in dataset if len(d['text']) > 20][:100]
baseline_loss = avg_loss(clean_model, tokenizer, texts)

results = []
for layer in range(12):
    print(f"\nEvaluating layer: {layer}")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    params = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and f'transformer.h.{layer}' in name:
            params.append(p.data)

    total_capacity = sum(p.numel() * N_LSB for p in params)
    print(f"Total embedding capacity: {total_capacity} bits")
    if len(BITS) > total_capacity:
        print(f"Skipping layer {layer}: insufficient capacity ({len(BITS)} bits required)")
        continue

    cursor = 0
    for p in params:
        cap = p.numel() * N_LSB
        take = min(cap, len(BITS) - cursor)
        embed_bits_LSB(p, BITS[cursor:cursor+take], n_lsb=N_LSB)
        cursor += take
        if cursor >= len(BITS):
            break

    print(f"Injected {cursor} / {len(BITS)} bits ({cursor / len(BITS) * 100:.2f}%)")
    loss_inj = avg_loss(model, tokenizer, texts)
    kl = kl_divergence(clean_model, model, tokenizer, texts)
    delta_loss = (loss_inj - baseline_loss) / baseline_loss * 100
    results.append((layer, round(delta_loss, 6), round(kl, 6)))

print(f"\nSingle Layer Sensitivity Results ({PAYLOAD_MB}MB, {N_LSB}-LSB):")
print(f"{'Layer':<6} {'Δ-Loss (%)':<12} {'KL Divergence'}")
for layer, dl, kl in results:
    print(f"{layer:<6} {dl:<12} {kl}")
