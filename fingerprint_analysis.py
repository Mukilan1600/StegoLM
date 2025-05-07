
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

def get_flat_weights(model):
    tensors = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            tensors.append(param.detach().flatten())
    return torch.cat(tensors).cpu()

def entropy(hist):
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def analyze_models(base_path="gpt2", injected_paths=[]):
    print("Loading clean model...")
    model_clean = GPT2LMHeadModel.from_pretrained(base_path)
    w_clean = get_flat_weights(model_clean).numpy()

    results = []
    plt.figure(figsize=(10, 6))
    bins = np.linspace(-0.2, 0.2, 100)

    hist_clean, _ = np.histogram(w_clean, bins=bins)
    plt.plot(bins[:-1], hist_clean, label="Clean", linewidth=2)

    for path in injected_paths:
        print(f"Loading: {path}")
        model_inj = GPT2LMHeadModel.from_pretrained(path)
        w_inj = get_flat_weights(model_inj).numpy()

        hist_inj, _ = np.histogram(w_inj, bins=bins)
        plt.plot(bins[:-1], hist_inj, label=os.path.basename(path))

        # Metric comparison
        mean = np.mean(w_inj)
        std = np.std(w_inj)
        ent = entropy(hist_inj)
        cosine_sim = np.dot(w_clean, w_inj) / (np.linalg.norm(w_clean) * np.linalg.norm(w_inj))
        results.append((os.path.basename(path), mean, std, ent, cosine_sim))

    plt.title("Weight Distribution Histograms")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("weight_histograms.png", dpi=300)

    # Print comparison table
    print("\nFingerprint Metrics vs. Clean:")
    print(f"{'Model':<20} {'Mean':>10} {'Std':>10} {'Entropy':>10} {'CosineSim':>12}")
    for label, mean, std, ent, cos in results:
        print(f"{label:<20} {mean:10.5f} {std:10.5f} {ent:10.5f} {cos:12.6f}")

if __name__ == "__main__":
    analyze_models(
        base_path="gpt2",
        injected_paths=[
            "./checkpoints/injected_15MB",
            "./checkpoints/injected_45MB",
            "./checkpoints/injected_90MB",
            "./checkpoints/injected_150MB"
        ]
    )
