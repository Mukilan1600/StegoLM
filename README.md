
# StegoLM: Covert Payload Embedding in LLM Weights

This repository contains the full experimental pipeline for the paper:  
**“StegoLM: Covert Payload Embedding in LLM Weights”**  
_(Mukilan I.K., 2025)_

We demonstrate how multi-megabyte binary payloads can be embedded in the weights of GPT-2 Small using least-significant-bit (LSB) steganography without significant degradation in model output or performance.

---

## Repository Structure

| Script                     | Description |
|---------------------------|-------------|
| `grid_inject_and_save.py` | Embeds payloads into GPT-2 Small weights across a grid of payload sizes and LSB depths, saving new model checkpoints. |
| `visual_output_comparison.py` | Runs side-by-side text generation using clean vs. injected models to verify semantic fidelity. |
| `layer_sensitivity.py`    | Measures loss and KL divergence when payloads are embedded in specific transformer layers to identify optimal slots. |
| `fingerprint_analysis.py` | Analyzes statistical differences (mean, std, entropy, cosine sim) between clean and injected weights to evaluate detectability. |

---

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.13  
- Transformers ≥ 4.35  
- Datasets, NumPy, Matplotlib  

Install dependencies:
```bash
pip install torch transformers datasets matplotlib numpy
```

---

## Reproducing Experiments

1. **Generate Injected Models:**
```bash
python grid_inject_and_save.py
```

2. **Compare Model Outputs:**
```bash
python visual_output_comparison.py
```

3. **Run Layer Sensitivity Analysis:**
```bash
python layer_sensitivity.py
```

4. **Evaluate Fingerprint Resistance:**
```bash
python fingerprint_analysis.py
```

---

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{mukilan2025stegolm,
  title={StegoLM: Covert Payload Embedding in LLM Weights},
  author={Mukilan I.K.},
  year={2025},
  note={Preprint},
  url={https://github.com/Mukilan1600/StegoLM}
}
```

---

## 📬 Contact

For questions or collaborations, feel free to reach out via GitHub Issues or [mukilan.ik2@gmail.com].
