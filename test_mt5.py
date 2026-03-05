# ============================================================
# TEST SCRIPT — Run inference on test CSVs
# ============================================================

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

MODEL_DIR = "./mt5_gender_inclusive"
TEST_DIR = "data/A/Test"
OUTPUT_DIR = "test_outputs"


def build_prompt(text):
    return f"rewrite: {text}"


def generate(text, model, tokenizer):
    inputs = tokenizer(build_prompt(text), return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=4,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📦 Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    for file in os.listdir(TEST_DIR):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(TEST_DIR, file)
        df = pd.read_csv(path)

        input_col = df.columns[0]

        outputs = []
        for text in df[input_col]:
            text = str(text).strip()
            if not text:
                outputs.append("")
                continue

            outputs.append(generate(text, model, tokenizer))

        df["output"] = outputs

        save_path = os.path.join(OUTPUT_DIR, file)
        df.to_csv(save_path, index=False)

        print("✅ Saved:", save_path)


if __name__ == "__main__":
    main()
