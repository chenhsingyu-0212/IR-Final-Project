#!/bin/bash

# IR Final Project - Run All Experiments Script
# This script runs all experiments sequentially

echo "=========================================="
echo "IR Final Project - Running All Experiments"
echo "=========================================="

# Set default parameters
N_SAMPLES=${1:-100}  # Default to 100 samples for testing, use "all" for full run
echo "Running with $N_SAMPLES samples per experiment"

# Ensure results directory exists
mkdir -p results

# Basic check for dataset
if [ ! -f "datasets/test.json" ]; then
    echo "⚠️ Warning: datasets/test.json not found. Make sure datasets are prepared (run datasets/split_dataset.py)."
fi

# Function to run experiment with timing
run_experiment() {
    local exp_name=$1
    local exp_file=$2
    local extra_args=$3

    echo ""
    echo "----------------------------------------"
    echo "Running $exp_name..."
    echo "----------------------------------------"

    local start_time=$(date +%s)

    if [ "$N_SAMPLES" = "all" ]; then
        python $exp_file $extra_args
    else
        python $exp_file -n $N_SAMPLES $extra_args
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "$exp_name completed in $duration seconds"
}

# Run experiments 1-4
run_experiment "Experiment 1 (Direct Answer)" "1_experiment.py"
run_experiment "Experiment 2 (Fixed RAG)" "2_experiment.py" "-r 3"
run_experiment "Experiment 3 (Logistic Confidence)" "3_experiment.py" "-c 0.7"
run_experiment "Experiment 4 (Calibrated Confidence)" "4_experiment.py" "-r 3 -t 0.7"

# Run Experiment 5 (Fine-tuned Logistic Confidence)
echo ""
echo "----------------------------------------"
echo "Running Experiment 5 (Fine-tuned Logistic Confidence)..."
echo "----------------------------------------"

echo "Step 5.1: Generating training data..."
python 5_step1_experiment.py -n 1000 -o results/exp5_confidence_train.jsonl

echo "Step 5.2: Fine-tuning model..."
python 5_step2_experiment.py -e 1 -b 2

echo "Step 5.3: Running inference..."
if [ "$N_SAMPLES" = "all" ]; then
    python 5_step3_experiment.py -d datasets/test.json -r 3
else
    python 5_step3_experiment.py -d datasets/test.json -n $N_SAMPLES -r 3
fi

# Run Experiment 6 (Fine-tuned Calibrated Confidence)
echo ""
echo "----------------------------------------"
echo "Running Experiment 6 (Fine-tuned Calibrated Confidence)..."
echo "----------------------------------------"

echo "Step 6.1: Generating training data..."
python 6_step1_experiment.py -n 1000 -o results/exp6_calibrated_confidence_train.jsonl

echo "Step 6.2: Fine-tuning model..."
python 6_step2_experiment.py -e 1 -b 2

echo "Step 6.3: Running inference..."
if [ "$N_SAMPLES" = "all" ]; then
    python 6_step3_experiment.py -r 3
else
    python 6_step3_experiment.py -n $N_SAMPLES -r 3
fi

echo ""
echo "=========================================="
echo "All 6 experiments completed!"
echo "Output files generated:"
echo "- results/exp1.json (Direct Answer)"
echo "- results/exp2.json (Fixed RAG)"
echo "- results/exp3.json (Logit-based Confidence)"
echo "- results/exp4.json (Calibrated Confidence)"
echo "- results/exp5.json (Fine-tuned Logistic)"
echo "- results/exp6.json (Fine-tuned Calibrated)"
echo ""
echo "Training data:"
echo "- results/exp5_confidence_train.jsonl"
echo "- results/exp6_calibrated_confidence_train.jsonl"
echo ""
echo "Fine-tuned models:"
echo "- results/exp5_finetuned_confidence/ (Logistic)"
echo "- results/exp6_finetuned_calibrated_confidence/ (Calibrated)"
echo "=========================================="