#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp5 fine-tuning: train Mistral to output calibrated confidence
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# ========== CONFIG ==========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = "exp5_confidence_train.jsonl"
OUTPUT_DIR = "exp5_finetuned_confidence"

# ========== LOAD DATA ==========
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # 避免 padding token 錯誤

def preprocess(ex):
    prompt = f"{ex['instruction']}\n{ex['input']}\nConfidence:"
    model_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(ex["output"], truncation=True, padding="max_length", max_length=16)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ========== LOAD MODEL ==========
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
# 部分模型無 pad_token_id，需設定
model.config.pad_token_id = tokenizer.eos_token_id

# ========== APPLY LoRA ==========
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# ========== TRAINING ==========
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",  # 避免自動使用 wandb
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Fine-tuned model saved to {OUTPUT_DIR}")
