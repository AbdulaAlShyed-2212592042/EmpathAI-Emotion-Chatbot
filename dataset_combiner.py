#!/usr/bin/env python3
"""
dataset_combiner.py

Combine all cleaned datasets into a single JSON file with unified format.
Uses the label mappings from dataset_mapping.py to include emotion names.

Output format:
{
    "metadata": {
        "total_entries": <count>,
        "datasets": {<dataset_name>: <count>, ...},
        "splits": {<split_name>: <count>, ...},
        "created_at": "<timestamp>"
    },
    "data": [
        {
            "text": "<text>",
            "emotion_labels": [<numeric_labels>],
            "emotion_names": ["<emotion_names>"],
            "dataset": "<dataset_name>",
            "split": "<split_name>",
            "original_data": {<original_entry>}
        },
        ...
    ]
}
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


# Import label mappings from dataset_mapping.py
LABEL_MAPPINGS = {
    'go_emotions': {
        'label_field': 'labels',
        'is_multilabel': True,
        'names': ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 
                 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
                 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 
                 'sadness', 'surprise', 'neutral']
    },
    'emotion': {
        'label_field': 'label',
        'is_multilabel': False,
        'names': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    },
    'tweet_eval_emotion': {
        'label_field': 'label',
        'is_multilabel': False,
        'names': ['anger', 'joy', 'optimism', 'sadness']
    },
    'imdb': {
        'label_field': 'label',
        'is_multilabel': False,
        'names': ['negative', 'positive']
    },
    'yelp_review_full': {
        'label_field': 'label',
        'is_multilabel': False,
        'names': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    }
}


def get_emotion_info(dataset_name: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract emotion labels and names from original data."""
    if dataset_name not in LABEL_MAPPINGS:
        return {
            'emotion_labels': [],
            'emotion_names': [],
            'valid': False
        }
    
    mapping = LABEL_MAPPINGS[dataset_name]
    label_field = mapping['label_field']
    names = mapping['names']
    is_multilabel = mapping['is_multilabel']
    
    if label_field not in original_data:
        return {
            'emotion_labels': [],
            'emotion_names': [],
            'valid': False
        }
    
    label_value = original_data[label_field]
    
    if is_multilabel:
        # Multi-label case (GoEmotions)
        if not isinstance(label_value, list):
            return {'emotion_labels': [], 'emotion_names': [], 'valid': False}
        
        emotion_labels = label_value
        emotion_names = []
        for label in emotion_labels:
            if 0 <= label < len(names):
                emotion_names.append(names[label])
            else:
                emotion_names.append(f"unknown_{label}")
        
        return {
            'emotion_labels': emotion_labels,
            'emotion_names': emotion_names,
            'valid': True
        }
    else:
        # Single-label case
        if not isinstance(label_value, int):
            return {'emotion_labels': [], 'emotion_names': [], 'valid': False}
        
        if 0 <= label_value < len(names):
            emotion_name = names[label_value]
        else:
            emotion_name = f"unknown_{label_value}"
        
        return {
            'emotion_labels': [label_value],
            'emotion_names': [emotion_name],
            'valid': True
        }


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data


def combine_datasets(dataset_dir: str = 'dataset_original', output_file: str = 'combined_dataset.json') -> Dict[str, Any]:
    """Combine all datasets into a single JSON structure."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Dataset directory '{dataset_dir}' not found!")
        return {}
    
    print(f"COMBINING DATASETS FROM: {dataset_path.absolute()}")
    print("=" * 60)
    
    combined_data = []
    metadata = {
        'total_entries': 0,
        'datasets': {},
        'splits': {},
        'created_at': datetime.now().isoformat()
    }
    
    # Process each dataset
    for dataset_folder in dataset_path.iterdir():
        if dataset_folder.is_dir():
            dataset_name = dataset_folder.name
            print(f"\nProcessing dataset: {dataset_name}")
            
            dataset_count = 0
            
            # Process each split file
            for split_file in dataset_folder.glob('*.jsonl'):
                split_name = split_file.stem
                print(f"  Loading {split_name}...")
                
                data = load_jsonl(str(split_file))
                split_count = 0
                
                for original_entry in data:
                    # Get text field (handle different field names)
                    text = original_entry.get('text', '')
                    if not text:
                        continue  # Skip entries without text
                    
                    # Get emotion information
                    emotion_info = get_emotion_info(dataset_name, original_entry)
                    
                    if not emotion_info['valid']:
                        continue  # Skip entries with invalid emotions
                    
                    # Create unified entry
                    unified_entry = {
                        'text': text,
                        'emotion_labels': emotion_info['emotion_labels'],
                        'emotion_names': emotion_info['emotion_names'],
                        'dataset': dataset_name,
                        'split': split_name,
                        'original_data': original_entry
                    }
                    
                    combined_data.append(unified_entry)
                    split_count += 1
                    dataset_count += 1
                
                # Update split counts
                if split_name in metadata['splits']:
                    metadata['splits'][split_name] += split_count
                else:
                    metadata['splits'][split_name] = split_count
                
                print(f"    Added {split_count} entries from {split_name}")
            
            # Update dataset counts
            metadata['datasets'][dataset_name] = dataset_count
            metadata['total_entries'] += dataset_count
            print(f"  Total from {dataset_name}: {dataset_count} entries")
    
    # Create final structure
    result = {
        'metadata': metadata,
        'data': combined_data
    }
    
    print(f"\n" + "=" * 60)
    print(f"COMBINATION SUMMARY")
    print(f"Total entries: {metadata['total_entries']:,}")
    print(f"Datasets: {len(metadata['datasets'])}")
    print(f"Splits: {len(metadata['splits'])}")
    
    print(f"\nDataset breakdown:")
    for dataset, count in metadata['datasets'].items():
        percentage = (count / metadata['total_entries'] * 100) if metadata['total_entries'] > 0 else 0
        print(f"  {dataset}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nSplit breakdown:")
    for split, count in metadata['splits'].items():
        percentage = (count / metadata['total_entries'] * 100) if metadata['total_entries'] > 0 else 0
        print(f"  {split}: {count:,} ({percentage:.1f}%)")
    
    # Save to file
    print(f"\nSaving combined dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Combined dataset saved successfully!")
    return result


def show_samples(combined_data: Dict[str, Any], num_samples: int = 5) -> None:
    """Show sample entries from the combined dataset."""
    if 'data' not in combined_data or not combined_data['data']:
        print("No data found in combined dataset!")
        return
    
    print(f"\n" + "=" * 60)
    print(f"SAMPLE ENTRIES ({num_samples} examples)")
    print("=" * 60)
    
    data = combined_data['data']
    samples = data[:num_samples] if len(data) >= num_samples else data
    
    for i, entry in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Dataset: {entry['dataset']}")
        print(f"  Split: {entry['split']}")
        print(f"  Text: {entry['text'][:100]}{'...' if len(entry['text']) > 100 else ''}")
        print(f"  Emotion Labels: {entry['emotion_labels']}")
        print(f"  Emotion Names: {entry['emotion_names']}")


def main():
    """Main function."""
    print("DATASET COMBINER")
    print("================")
    print("This tool combines all cleaned datasets into a single JSON file.\n")
    
    # Combine datasets
    result = combine_datasets()
    
    if result:
        # Show samples
        show_samples(result)
        
        print(f"\n" + "=" * 60)
        print("COMBINATION COMPLETE!")
        print("=" * 60)
        print("The combined dataset includes:")
        print("- Unified text field")
        print("- Numeric emotion labels")
        print("- Human-readable emotion names")
        print("- Dataset source and split information")
        print("- Original data preserved")
        print("\nUse 'combined_dataset.json' for training or analysis!")
    else:
        print("Failed to combine datasets!")


if __name__ == "__main__":
    main()