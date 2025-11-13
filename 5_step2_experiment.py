#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 5 Step 2: Fine-tuned Confidence (Logistic)
Description: LoRA-fine-tuned model to predict confidence from average token probability and perplexity
Retrieval Trigger: Learned decision
- Step 2: Fine-tune the model using LoRA

Usage:
    python 5_step2_experiment.py                    # Fine-tune with default settings
    python 5_step2_experiment.py -e 2              # Train for 2 epochs
    python 5_step2_experiment.py -b 4              # Use batch size 4
    python 5_step2_experiment.py -o custom_model   # Custom output directory
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral model for confidence calibration")
    parser.add_argument("-m", "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="Pre-trained model name or path")
    parser.add_argument("-d", "--data_path", type=str, default="results/exp5_confidence_train.jsonl",
                       help="Path to training data JSONL file")
    parser.add_argument("-o", "--output_dir", type=str, default="results/exp5_finetuned_confidence",
                       help="Output directory for fine-tuned model")
    parser.add_argument("-e", "--num_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2,
                       help="Per-device train batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=2e-5,
                       help="Learning rate for training")
    
    args = parser.parse_args()
    
    # ========== LOAD DATA ==========
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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
        args.model_name,
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
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",  # 避免自動使用 wandb
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Fine-tuned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
