# ============================================================
# TRAINING SCRIPT — Multilingual Gender Inclusive mT5
# ============================================================

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

BASE_MODEL = "google/mt5-small"
OUTPUT_DIR = "./mt5_gender_inclusive"
DATA_PATH = "data/A/Train"


# ------------------------------------------------------------
# NORMALIZE COLUMNS
# ------------------------------------------------------------
def normalize_columns(df):
    df = df.copy()

    # make columns lowercase for matching
    df.columns = [c.strip().lower() for c in df.columns]

    biased_col = None
    inclusive_col = None

    # 🔍 try keyword matching
    for col in df.columns:
        if biased_col is None and any(k in col for k in ["non", "biased"]):
            biased_col = col
        if inclusive_col is None and any(k in col for k in ["inclusive", "neutral"]):
            inclusive_col = col

    # 🔁 fallback → first two text columns
    if biased_col is None or inclusive_col is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]

        if len(text_cols) < 2:
            raise ValueError(f"Could not detect sentence columns. Found: {df.columns}")

        biased_col, inclusive_col = text_cols[:2]

    print(f"➡ detected columns → biased='{biased_col}', inclusive='{inclusive_col}'")

    df = df.rename(columns={
        biased_col: "biased",
        inclusive_col: "inclusive"
    })

    return df[["biased", "inclusive"]]



# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
def load_data():
    language_map = {
        "English": "en",
        "German": "de",
        "Spanish": "es",
        "Tamil": "ta",
        "Kannada": "kn"
    }

    dfs = []

    for folder, lang in language_map.items():
        path = os.path.join(DATA_PATH, folder, "SentencePairs.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path).dropna()
        df = normalize_columns(df)
        df["language"] = lang
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------
def build_prompt(text):
    return f"rewrite: {text}"


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
def prepare_dataset(df):
    rows = []

    for _, row in df.iterrows():
        rows.append({
            "input_text": build_prompt(row["biased"]),
            "target_text": row["inclusive"]
        })

    return Dataset.from_list(rows)


# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------
def tokenize(batch, tokenizer):
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=96)
    targets = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=96)

    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in targets["input_ids"]
    ]

    inputs["labels"] = labels
    return inputs


# ------------------------------------------------------------
# MAIN TRAIN
# ------------------------------------------------------------
def main():
    print("🚀 Loading data...")
    df = load_data()
    dataset = prepare_dataset(df)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    tokenized_ds = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=3e-5,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    trainer.train()

    # 🔴 IMPORTANT → save final model (NOT checkpoint)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Training complete. Model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
