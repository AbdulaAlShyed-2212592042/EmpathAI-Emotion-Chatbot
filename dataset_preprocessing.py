#!/usr/bin/env python3
"""
dataset_preprocessing.py

Preprocess the combined emotion dataset for RoBERTa-base-sentiment training.
This script loads the combined dataset, applies text preprocessing suitable for RoBERTa,
creates proper train/validation/test splits, and saves the preprocessed data in multiple formats
suitable for Hugging Face transformers training.

Features:
- Text cleaning and normalization for RoBERTa
- Proper tokenization handling
- Label encoding and mapping
- Train/validation/test splits
- Multiple output formats (JSON, CSV, HuggingFace datasets)
- Data statistics and quality checks
"""

import json
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
    import torch
except ImportError as e:
    logger.warning(f"Some packages not installed: {e}")
    logger.warning("Please install: pip install transformers datasets scikit-learn torch")


class DatasetPreprocessor:
    """
    Main class for preprocessing emotion datasets for RoBERTa training.
    """
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 max_length: int = 512,
                 output_dir: str = "preprocessed_data"):
        """
        Initialize the preprocessor.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for tokenization
            output_dir: Directory to save preprocessed data
        """
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer for {model_name}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # Dataset and label information
        self.data = None
        self.processed_data = None
        self.label_encoders = {}
        self.emotion_mapping = {}
        self.stats = {}
    
    def load_combined_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load the combined emotion dataset.
        
        Args:
            dataset_path: Path to combined_dataset_clean.json
            
        Returns:
            Loaded dataset dictionary
        """
        logger.info(f"Loading combined dataset from {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded dataset with {data['metadata']['total_entries']} entries")
            logger.info(f"Datasets included: {list(data['metadata']['datasets'].keys())}")
            
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for RoBERTa processing.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with tokenization
        # But keep punctuation as it's important for sentiment
        text = re.sub(r'[^\w\s\.,!?;:\'"()#@-]', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Handle URLs and mentions (common in Twitter data)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        text = re.sub(r'@\w+', '[MENTION]', text)
        text = re.sub(r'#\w+', '[HASHTAG]', text)
        
        # Remove excessive repetitions
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        return text.strip()
    
    def create_unified_labels(self) -> Dict[str, Any]:
        """
        Create unified emotion labels across all datasets.
        
        Returns:
            Dictionary with label mappings and encoders
        """
        logger.info("Creating unified emotion labels")
        
        if not self.data:
            raise ValueError("No data loaded. Call load_combined_dataset first.")
        
        # Collect all unique emotion names
        all_emotions = set()
        emotion_freq = Counter()
        
        for entry in self.data['data']:
            emotions = entry['emotion_names']
            if isinstance(emotions, list):
                for emotion in emotions:
                    all_emotions.add(emotion)
                    emotion_freq[emotion] += 1
            else:
                all_emotions.add(emotions)
                emotion_freq[emotions] += 1
        
        # Sort emotions by frequency
        sorted_emotions = [emotion for emotion, _ in emotion_freq.most_common()]
        
        # Create label mappings
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(sorted_emotions)}
        id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}
        
        self.emotion_mapping = {
            'emotion_to_id': emotion_to_id,
            'id_to_emotion': id_to_emotion,
            'all_emotions': sorted_emotions,
            'emotion_freq': dict(emotion_freq)
        }
        
        logger.info(f"Created unified label mapping with {len(sorted_emotions)} emotions")
        logger.info(f"Most common emotions: {sorted_emotions[:10]}")
        
        return self.emotion_mapping
    
    def preprocess_entries(self) -> List[Dict[str, Any]]:
        """
        Preprocess all dataset entries.
        
        Returns:
            List of preprocessed entries
        """
        logger.info("Preprocessing dataset entries")
        
        if not self.data:
            raise ValueError("No data loaded. Call load_combined_dataset first.")
        
        if not self.emotion_mapping:
            self.create_unified_labels()
        
        processed_entries = []
        skipped_count = 0
        
        for entry in self.data['data']:
            try:
                # Clean text
                cleaned_text = self.clean_text(entry['text'])
                
                # Skip empty texts
                if not cleaned_text or len(cleaned_text.strip()) < 3:
                    skipped_count += 1
                    continue
                
                # Convert emotion names to IDs
                emotion_names = entry['emotion_names']
                if isinstance(emotion_names, str):
                    emotion_names = [emotion_names]
                
                emotion_ids = []
                for emotion in emotion_names:
                    if emotion in self.emotion_mapping['emotion_to_id']:
                        emotion_ids.append(self.emotion_mapping['emotion_to_id'][emotion])
                
                # Skip entries without valid emotions
                if not emotion_ids:
                    skipped_count += 1
                    continue
                
                # Create processed entry
                processed_entry = {
                    'text': cleaned_text,
                    'emotion_ids': emotion_ids,
                    'emotion_names': emotion_names,
                    'dataset_source': entry['dataset'],
                    'original_split': entry['split'],
                    'is_multilabel': len(emotion_ids) > 1,
                    'text_length': len(cleaned_text),
                    'word_count': len(cleaned_text.split())
                }
                
                # Add tokenization if tokenizer is available
                if self.tokenizer:
                    tokens = self.tokenizer(
                        cleaned_text,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors=None
                    )
                    processed_entry['token_count'] = len(tokens['input_ids'])
                    processed_entry['is_truncated'] = len(tokens['input_ids']) >= self.max_length
                
                processed_entries.append(processed_entry)
                
            except Exception as e:
                logger.warning(f"Error processing entry: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Processed {len(processed_entries)} entries, skipped {skipped_count}")
        
        self.processed_data = processed_entries
        return processed_entries
    
    def create_train_val_test_splits(self, 
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   stratify_by: str = 'primary_emotion',
                                   random_seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create train/validation/test splits.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify_by: How to stratify the splits
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Creating train/validation/test splits")
        
        if not self.processed_data:
            raise ValueError("No processed data. Call preprocess_entries first.")
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Create stratification labels based on primary emotion
        stratify_labels = []
        for entry in self.processed_data:
            # Use the first (most prominent) emotion for stratification
            primary_emotion = entry['emotion_ids'][0] if entry['emotion_ids'] else -1
            stratify_labels.append(primary_emotion)
        
        # First split: train + val vs test
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            self.processed_data,
            stratify_labels,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=stratify_labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_data, val_data, _, _ = train_test_split(
            train_val_data,
            train_val_labels,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=train_val_labels
        )
        
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return splits
    
    def compute_statistics(self, splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for the processed dataset.
        
        Args:
            splits: Train/val/test splits
            
        Returns:
            Statistics dictionary
        """
        logger.info("Computing dataset statistics")
        
        stats = {
            'general': {},
            'splits': {},
            'emotions': {},
            'text_properties': {},
            'dataset_distribution': {}
        }
        
        # General statistics
        total_entries = sum(len(split_data) for split_data in splits.values())
        stats['general'] = {
            'total_entries': total_entries,
            'num_emotions': len(self.emotion_mapping['all_emotions']),
            'split_sizes': {split_name: len(split_data) for split_name, split_data in splits.items()}
        }
        
        # Per-split statistics
        for split_name, split_data in splits.items():
            split_stats = {
                'size': len(split_data),
                'multilabel_ratio': sum(1 for entry in split_data if entry['is_multilabel']) / len(split_data),
                'avg_text_length': np.mean([entry['text_length'] for entry in split_data]),
                'avg_word_count': np.mean([entry['word_count'] for entry in split_data]),
            }
            
            if self.tokenizer:
                split_stats['avg_token_count'] = np.mean([entry.get('token_count', 0) for entry in split_data])
                split_stats['truncation_ratio'] = sum(1 for entry in split_data if entry.get('is_truncated', False)) / len(split_data)
            
            stats['splits'][split_name] = split_stats
        
        # Emotion distribution
        emotion_counts = defaultdict(int)
        for split_data in splits.values():
            for entry in split_data:
                for emotion_id in entry['emotion_ids']:
                    emotion_name = self.emotion_mapping['id_to_emotion'][emotion_id]
                    emotion_counts[emotion_name] += 1
        
        stats['emotions'] = {
            'distribution': dict(emotion_counts),
            'most_common': sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'least_common': sorted(emotion_counts.items(), key=lambda x: x[1])[:10]
        }
        
        # Dataset source distribution
        dataset_counts = defaultdict(int)
        for split_data in splits.values():
            for entry in split_data:
                dataset_counts[entry['dataset_source']] += 1
        
        stats['dataset_distribution'] = dict(dataset_counts)
        
        self.stats = stats
        return stats
    
    def save_preprocessed_data(self, splits: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Save preprocessed data in multiple formats.
        
        Args:
            splits: Train/val/test splits to save
        """
        logger.info(f"Saving preprocessed data to {self.output_dir}")
        
        # Create subdirectories
        (self.output_dir / 'json').mkdir(exist_ok=True)
        (self.output_dir / 'csv').mkdir(exist_ok=True)
        (self.output_dir / 'huggingface').mkdir(exist_ok=True)
        
        # Save as JSON files
        for split_name, split_data in splits.items():
            json_path = self.output_dir / 'json' / f'{split_name}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {split_name} split as JSON: {json_path}")
        
        # Save as CSV files
        for split_name, split_data in splits.items():
            # Flatten the data for CSV
            csv_data = []
            for entry in split_data:
                csv_row = {
                    'text': entry['text'],
                    'emotion_ids': ','.join(map(str, entry['emotion_ids'])),
                    'emotion_names': ','.join(entry['emotion_names']),
                    'dataset_source': entry['dataset_source'],
                    'original_split': entry['original_split'],
                    'is_multilabel': entry['is_multilabel'],
                    'text_length': entry['text_length'],
                    'word_count': entry['word_count']
                }
                
                if 'token_count' in entry:
                    csv_row['token_count'] = entry['token_count']
                    csv_row['is_truncated'] = entry['is_truncated']
                
                csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / 'csv' / f'{split_name}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Saved {split_name} split as CSV: {csv_path}")
        
        # Save as HuggingFace Dataset
        try:
            if Dataset is not None:
                hf_datasets = {}
                for split_name, split_data in splits.items():
                    # Prepare data for HuggingFace format
                    hf_data = {
                        'text': [entry['text'] for entry in split_data],
                        'labels': [entry['emotion_ids'] for entry in split_data],
                        'emotion_names': [entry['emotion_names'] for entry in split_data],
                        'dataset_source': [entry['dataset_source'] for entry in split_data]
                    }
                    
                    hf_datasets[split_name] = Dataset.from_dict(hf_data)
                
                dataset_dict = DatasetDict(hf_datasets)
                hf_path = self.output_dir / 'huggingface'
                dataset_dict.save_to_disk(str(hf_path))
                logger.info(f"Saved as HuggingFace dataset: {hf_path}")
                
        except Exception as e:
            logger.warning(f"Could not save HuggingFace format: {e}")
        
        # Save metadata and mappings
        metadata = {
            'preprocessing_info': {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'created_at': datetime.now().isoformat(),
                'total_entries': sum(len(split_data) for split_data in splits.values())
            },
            'emotion_mapping': self.emotion_mapping,
            'statistics': self.stats
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata: {metadata_path}")
    
    def create_roberta_training_format(self, splits: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Create training data specifically formatted for RoBERTa fine-tuning.
        
        Args:
            splits: Train/val/test splits
        """
        logger.info("Creating RoBERTa-specific training format")
        
        roberta_dir = self.output_dir / 'roberta_training'
        roberta_dir.mkdir(exist_ok=True)
        
        # Create label mappings for different training scenarios
        label_mappings = {
            'multi_emotion': self.emotion_mapping,
            'binary_sentiment': {
                'positive_emotions': ['joy', 'love', 'admiration', 'amusement', 'approval', 'caring', 'excitement', 'gratitude', 'optimism', 'pride', 'relief'],
                'negative_emotions': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'],
                'neutral_emotions': ['neutral', 'confusion', 'curiosity', 'realization', 'surprise']
            }
        }
        
        for split_name, split_data in splits.items():
            # Multi-emotion classification format
            multi_emotion_data = []
            for entry in split_data:
                multi_emotion_data.append({
                    'text': entry['text'],
                    'labels': entry['emotion_ids'],  # List of emotion IDs for multi-label
                    'label_names': entry['emotion_names']
                })
            
            # Binary sentiment format
            binary_sentiment_data = []
            for entry in split_data:
                # Convert emotions to sentiment
                sentiment_score = 0
                for emotion_name in entry['emotion_names']:
                    if emotion_name in label_mappings['binary_sentiment']['positive_emotions']:
                        sentiment_score += 1
                    elif emotion_name in label_mappings['binary_sentiment']['negative_emotions']:
                        sentiment_score -= 1
                
                # Determine binary sentiment
                if sentiment_score > 0:
                    binary_label = 1  # Positive
                    binary_name = 'positive'
                elif sentiment_score < 0:
                    binary_label = 0  # Negative
                    binary_name = 'negative'
                else:
                    binary_label = 2  # Neutral
                    binary_name = 'neutral'
                
                binary_sentiment_data.append({
                    'text': entry['text'],
                    'label': binary_label,
                    'label_name': binary_name
                })
            
            # Save both formats
            multi_path = roberta_dir / f'{split_name}_multi_emotion.json'
            with open(multi_path, 'w', encoding='utf-8') as f:
                json.dump(multi_emotion_data, f, indent=2, ensure_ascii=False)
            
            binary_path = roberta_dir / f'{split_name}_binary_sentiment.json'
            with open(binary_path, 'w', encoding='utf-8') as f:
                json.dump(binary_sentiment_data, f, indent=2, ensure_ascii=False)
        
        # Save label mappings
        mappings_path = roberta_dir / 'label_mappings.json'
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(label_mappings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created RoBERTa training formats in {roberta_dir}")
    
    def run_full_preprocessing(self, 
                             dataset_path: str = 'combined_dataset_clean.json',
                             **kwargs) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            dataset_path: Path to the combined dataset
            **kwargs: Additional arguments for split creation
            
        Returns:
            Complete preprocessing results
        """
        logger.info("Starting full preprocessing pipeline")
        
        # Load data
        self.load_combined_dataset(dataset_path)
        
        # Preprocess entries
        self.preprocess_entries()
        
        # Create splits
        splits = self.create_train_val_test_splits(**kwargs)
        
        # Compute statistics
        stats = self.compute_statistics(splits)
        
        # Save all formats
        self.save_preprocessed_data(splits)
        
        # Create RoBERTa-specific format
        self.create_roberta_training_format(splits)
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return {
            'splits': splits,
            'statistics': stats,
            'emotion_mapping': self.emotion_mapping,
            'output_dir': str(self.output_dir)
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess emotion dataset for RoBERTa training')
    parser.add_argument('--input', '-i', type=str, default='combined_dataset_clean.json',
                       help='Path to combined dataset JSON file')
    parser.add_argument('--output', '-o', type=str, default='preprocessed_data',
                       help='Output directory for preprocessed data')
    parser.add_argument('--model', '-m', type=str, default='roberta-base',
                       help='Model name for tokenizer')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(
        model_name=args.model,
        max_length=args.max_length,
        output_dir=args.output
    )
    
    # Run preprocessing
    results = preprocessor.run_full_preprocessing(
        dataset_path=args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Input dataset: {args.input}")
    print(f"Output directory: {results['output_dir']}")
    print(f"Total entries processed: {results['statistics']['general']['total_entries']:,}")
    print(f"Number of emotions: {results['statistics']['general']['num_emotions']}")
    print(f"Split sizes: {results['statistics']['general']['split_sizes']}")
    print("\nTop 10 emotions:")
    for emotion, count in results['statistics']['emotions']['most_common']:
        print(f"  {emotion}: {count:,}")
    
    print(f"\nPreprocessed data saved to: {results['output_dir']}")
    print("Available formats:")
    print("  - JSON files (json/)")
    print("  - CSV files (csv/)")
    print("  - HuggingFace datasets (huggingface/)")
    print("  - RoBERTa training format (roberta_training/)")
    print("  - Metadata and statistics (metadata.json)")


if __name__ == "__main__":
    main()