from test_mt5 import generate
from transformers import T5ForConditionalGeneration, AutoTokenizer

MODEL_DIR = "mt5_gender_inclusive"

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

print(
    generate_output(
        "Each salesman must submit his report.",
        "en",
        model,
        tokenizer
    )
)
