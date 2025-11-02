#!/usr/bin/env python3
"""
validate_preprocessed_data.py

Validate and demonstrate the preprocessed emotion dataset for RoBERTa training.
This script loads the preprocessed data and shows samples, statistics, and 
verifies the data quality for training.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np

def load_and_validate_data(data_dir: str = "preprocessed_data_roberta"):
    """Load and validate the preprocessed data."""
    data_path = Path(data_dir)
    
    print("=" * 60)
    print("PREPROCESSED DATA VALIDATION")
    print("=" * 60)
    
    # Load metadata
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print("METADATA SUMMARY:")
        print(f"Model: {metadata['preprocessing_info']['model_name']}")
        print(f"Max length: {metadata['preprocessing_info']['max_length']}")
        print(f"Total entries: {metadata['preprocessing_info']['total_entries']:,}")
        print(f"Created: {metadata['preprocessing_info']['created_at']}")
        print(f"Number of emotions: {len(metadata['emotion_mapping']['all_emotions'])}")
        
        # Show emotion distribution
        print(f"\nTOP 15 EMOTIONS:")
        for emotion, count in sorted(metadata['statistics']['emotions']['distribution'].items(), 
                                   key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {emotion}: {count:,}")
        
        print(f"\nSPLIT STATISTICS:")
        for split_name, stats in metadata['statistics']['splits'].items():
            print(f"  {split_name}:")
            print(f"    Size: {stats['size']:,}")
            print(f"    Multilabel ratio: {stats['multilabel_ratio']:.1%}")
            print(f"    Avg text length: {stats['avg_text_length']:.1f}")
            print(f"    Avg word count: {stats['avg_word_count']:.1f}")
            if 'avg_token_count' in stats:
                print(f"    Avg token count: {stats['avg_token_count']:.1f}")
                print(f"    Truncation ratio: {stats['truncation_ratio']:.1%}")
    
    # Validate JSON files
    print(f"\nJSON FILES VALIDATION:")
    json_dir = data_path / "json"
    for split_file in ["train.json", "validation.json", "test.json"]:
        file_path = json_dir / split_file
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  {split_file}: {len(data):,} entries")
            
            # Show sample
            if data:
                sample = data[0]
                print(f"    Sample text: '{sample['text'][:100]}...'")
                print(f"    Emotion IDs: {sample['emotion_ids']}")
                print(f"    Emotion names: {sample['emotion_names']}")
                print(f"    Dataset source: {sample['dataset_source']}")
    
    # Validate CSV files
    print(f"\nCSV FILES VALIDATION:")
    csv_dir = data_path / "csv"
    for split_file in ["train.csv", "validation.csv", "test.csv"]:
        file_path = csv_dir / split_file
        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"  {split_file}: {len(df):,} rows, {len(df.columns)} columns")
            print(f"    Columns: {list(df.columns)}")
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                print(f"    Missing values: {missing[missing > 0].to_dict()}")
            else:
                print(f"    No missing values")
    
    # Validate RoBERTa training files
    print(f"\nROBERTA TRAINING FILES:")
    roberta_dir = data_path / "roberta_training"
    if roberta_dir.exists():
        for file in roberta_dir.glob("*.json"):
            if file.name == "label_mappings.json":
                with open(file, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                print(f"  {file.name}: Contains label mappings")
                print(f"    Multi-emotion mapping: {len(mappings['multi_emotion']['all_emotions'])} emotions")
                print(f"    Binary sentiment mapping: 3 categories (pos/neg/neutral)")
            else:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  {file.name}: {len(data):,} entries")
                
                if data:
                    sample = data[0]
                    print(f"    Sample: '{sample['text'][:50]}...'")
                    if 'labels' in sample:  # Multi-emotion format
                        print(f"    Labels: {sample['labels']} -> {sample['label_names']}")
                    elif 'label' in sample:  # Binary sentiment format
                        print(f"    Label: {sample['label']} -> {sample['label_name']}")

def demonstrate_training_readiness(data_dir: str = "preprocessed_data_roberta"):
    """Demonstrate that the data is ready for RoBERTa training."""
    print(f"\n" + "=" * 60)
    print("TRAINING READINESS DEMONSTRATION")
    print("=" * 60)
    
    data_path = Path(data_dir)
    
    # Load a sample of training data
    train_file = data_path / "roberta_training" / "train_multi_emotion.json"
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        print(f"MULTI-EMOTION CLASSIFICATION DATA:")
        print(f"Training samples: {len(train_data):,}")
        
        # Show label distribution
        all_labels = []
        for sample in train_data[:10000]:  # Sample for faster processing
            all_labels.extend(sample['labels'])
        
        label_counts = Counter(all_labels)
        print(f"Label distribution (top 10):")
        for label_id, count in label_counts.most_common(10):
            print(f"  Label {label_id}: {count:,} occurrences")
        
        print(f"\nSample training examples:")
        for i, sample in enumerate(train_data[:3]):
            print(f"  Example {i+1}:")
            print(f"    Text: '{sample['text'][:80]}...'")
            print(f"    Labels: {sample['labels']} -> {sample['label_names']}")
    
    # Binary sentiment data
    binary_file = data_path / "roberta_training" / "train_binary_sentiment.json"
    if binary_file.exists():
        with open(binary_file, 'r', encoding='utf-8') as f:
            binary_data = json.load(f)
        
        print(f"\nBINARY SENTIMENT CLASSIFICATION DATA:")
        print(f"Training samples: {len(binary_data):,}")
        
        # Show label distribution
        label_counts = Counter(sample['label'] for sample in binary_data)
        label_names = Counter(sample['label_name'] for sample in binary_data)
        
        print(f"Sentiment distribution:")
        for (label_id, count), (label_name, _) in zip(label_counts.most_common(), label_names.most_common()):
            print(f"  {label_name} (ID {label_id}): {count:,} samples")
        
        print(f"\nSample training examples:")
        for i, sample in enumerate(binary_data[:3]):
            print(f"  Example {i+1}:")
            print(f"    Text: '{sample['text'][:80]}...'")
            print(f"    Sentiment: {sample['label']} -> {sample['label_name']}")

def show_huggingface_format(data_dir: str = "preprocessed_data_roberta"):
    """Show how to load the HuggingFace format data."""
    print(f"\n" + "=" * 60)
    print("HUGGINGFACE DATASET FORMAT")
    print("=" * 60)
    
    try:
        from datasets import load_from_disk
        
        hf_path = Path(data_dir) / "huggingface"
        if hf_path.exists():
            dataset = load_from_disk(str(hf_path))
            
            print(f"Dataset loaded successfully!")
            print(f"Splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                print(f"  {split_name}: {len(split_data):,} examples")
                print(f"    Features: {list(split_data.features.keys())}")
            
            # Show a sample
            train_sample = dataset['train'][0]
            print(f"\nSample from training set:")
            print(f"  Text: '{train_sample['text'][:100]}...'")
            print(f"  Labels: {train_sample['labels']}")
            print(f"  Emotion names: {train_sample['emotion_names']}")
            print(f"  Dataset source: {train_sample['dataset_source']}")
            
            print(f"\nTo use this dataset in your training:")
            print(f"```python")
            print(f"from datasets import load_from_disk")
            print(f"dataset = load_from_disk('{hf_path}')")
            print(f"train_dataset = dataset['train']")
            print(f"val_dataset = dataset['validation']")
            print(f"test_dataset = dataset['test']")
            print(f"```")
        
    except ImportError:
        print("datasets library not available for demonstration")
    except Exception as e:
        print(f"Error loading HuggingFace dataset: {e}")

def create_training_script_template(data_dir: str = "preprocessed_data_roberta"):
    """Create a template training script for RoBERTa."""
    template = '''#!/usr/bin/env python3
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
'''
    
    output_file = Path(data_dir).parent / "train_roberta_template.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"\nCreated training script template: {output_file}")
    print(f"Modify this script according to your specific training needs.")

def main():
    """Main validation function."""
    data_dir = "preprocessed_data_roberta"
    
    # Validate the preprocessed data
    load_and_validate_data(data_dir)
    
    # Demonstrate training readiness
    demonstrate_training_readiness(data_dir)
    
    # Show HuggingFace format usage
    show_huggingface_format(data_dir)
    
    # Create training script template
    create_training_script_template(data_dir)
    
    print(f"\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("Your preprocessed dataset is ready for RoBERTa training!")
    print(f"\nKey files for training:")
    print(f"  - Multi-emotion: {data_dir}/roberta_training/train_multi_emotion.json")
    print(f"  - Binary sentiment: {data_dir}/roberta_training/train_binary_sentiment.json")
    print(f"  - HuggingFace format: {data_dir}/huggingface/")
    print(f"  - CSV format: {data_dir}/csv/")
    print(f"  - Metadata: {data_dir}/metadata.json")

if __name__ == "__main__":
    main()