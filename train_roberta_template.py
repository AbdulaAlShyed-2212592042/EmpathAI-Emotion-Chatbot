#!/usr/bin/env python3
"""
train_roberta_sentiment.py

Template script for training RoBERTa on the preprocessed emotion dataset.
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_preprocessed_data(data_dir: str = "preprocessed_data_roberta"):
    """Load the preprocessed dataset."""
    # Option 1: Load HuggingFace format
    dataset = load_from_disk(f"{data_dir}/huggingface")
    
    # Option 2: Load JSON format
    # with open(f"{data_dir}/roberta_training/train_binary_sentiment.json", 'r') as f:
    #     train_data = json.load(f)
    
    return dataset

def setup_model_and_tokenizer(model_name: str = "roberta-base", num_labels: int = 3):
    """Set up the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    return model, tokenizer

def preprocess_function(examples, tokenizer, max_length: int = 512):
    """Tokenize the examples."""
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding=True,
        max_length=max_length
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    # Load data
    dataset = load_preprocessed_data()
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenize data
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test Results: {eval_results}")
    
    # Save model
    trainer.save_model("./trained_roberta_sentiment")

if __name__ == "__main__":
    main()
