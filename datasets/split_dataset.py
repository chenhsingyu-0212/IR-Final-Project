#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split single_answer_qa.json into train.json (80%) and test.json (20%).
Shuffle the dataset before splitting.
"""

import json
import random

DATA_FILE = "single_answer_qa.json"
TRAIN_FILE = "train.json"
TEST_FILE = "test.json"
TRAIN_RATIO = 0.75

# Load data
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle
random.shuffle(data)

# Split
total = len(data)
train_size = int(total * TRAIN_RATIO)
train_data = data[:train_size]
test_data = data[train_size:]

# Save
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(TEST_FILE, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Split complete: {len(train_data)} train samples, {len(test_data)} test samples")
print(f"Saved to {TRAIN_FILE} and {TEST_FILE}")