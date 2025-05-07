# visual_output_comparison.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(model, tokenizer, prompt, temperature=0.8):
    # Tokenize with explicit pad_token & attention_mask
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,           # now safe—pad_token is defined
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    # Generate using the attention mask
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=False,      # deterministic
        top_p=1.0,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_model(path_or_name):
    model = GPT2LMHeadModel.from_pretrained(path_or_name)
    # Make sure model knows its pad token
    model.config.pad_token_id = model.config.eos_token_id
    return model.eval()

def main():

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Define pad_token → EOS so padding works
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = "The future of cybersecurity depends on"
    model_paths = {
        "Clean": "gpt2",
        "Injected (15MB)": "./checkpoints/injected_15MB",
        "Injected (45MB)": "./checkpoints/injected_45MB",
        "Injected (90MB)": "./checkpoints/injected_90MB",
        "Injected (150MB)": "./checkpoints/injected_150MB"
    }

    print(f"Prompt: \"{prompt}\"\n")
    for label, path in model_paths.items():
        model = load_model(path)
        print("Loaded model...")
        text = generate_text(model, tokenizer, prompt)
        del model
        print(f"{label}:\n{text}\n{'-'*80}")

if __name__ == "__main__":
    main()

