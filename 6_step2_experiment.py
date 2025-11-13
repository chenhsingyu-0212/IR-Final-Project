#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 6 Step 2: Fine-tuned Confidence (Calibrated)
Description: LoRA fine-tune model to predict calibrated confidence
Retrieval Trigger: Learned decision
- Step 2: Fine-tune the model using LoRA for calibrated confidence prediction

Usage:
    python 6_step2_experiment.py                    # Fine-tune with default settings
    python 6_step2_experiment.py -e 2              # Train for 2 epochs
    python 6_step2_experiment.py -b 4              # Use batch size 4
    python 6_step2_experiment.py -o custom_model   # Custom output directory
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral model for calibrated confidence prediction")
    parser.add_argument("-m", "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="Pre-trained model name or path")
    parser.add_argument("-d", "--data_path", type=str, default="results/exp6_calibrated_confidence_train.jsonl",
                       help="Path to training data JSONL file")
    parser.add_argument("-o", "--output_dir", type=str, default="results/exp6_finetuned_calibrated_confidence",
                       help="Output directory for fine-tuned model")
    parser.add_argument("-e", "--num_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2,
                       help="Per-device train batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=2e-5,
                       help="Learning rate for training")

    args = parser.parse_args()

    print(f"Fine-tuning model for calibrated confidence prediction...")
    print(f"  Model: {args.model_name}")
    print(f"  Data: {args.data_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")

    # ========== LOAD DATA ==========
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess(ex):
        prompt = f"{ex['instruction']}\n{ex['input']}\nConfidence:"
        full_text = f"{prompt}{ex['output']}"
        
        # Tokenize the full sequence
        tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
        
        # Create labels: -100 for prompt part, actual tokens for confidence part
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_tokens = tokenized["input_ids"]
        
        labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]
        labels = labels[:512]  # Truncate to max_length
        labels += [-100] * (512 - len(labels))  # Pad with -100
        
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # ========== LOAD MODEL ==========
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ========== TRAINING ==========
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Fine-tuned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()