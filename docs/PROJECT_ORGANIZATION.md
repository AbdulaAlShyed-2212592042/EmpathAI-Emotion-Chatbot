# Project Organization

This document describes the updated project structure.

## Directory Structure

```
EmpathAI-Emotion-Chatbot/
├── README.md                       # Main project documentation
├── requirements.txt                # Python dependencies
├── train_emotion_chatbot_roberta.py # Main training script
├── .gitignore                      # Git ignore rules
│
├── checkpoint/                     # Model checkpoints (git-ignored)
│   ├── best_model_loss.pt         # Best model by validation loss
│   ├── best_model_f1.pt           # Best model by F1 score
│   ├── checkpoint_latest.pt       # Latest checkpoint
│   └── checkpoint_epoch_*.pt      # Periodic checkpoints
│
├── data/                           # Dataset files
│   ├── combined_dataset_clean.json         # Unified dataset
│   └── preprocessed_data_roberta/          # Preprocessed training data
│       ├── json/                           # JSON format
│       ├── csv/                            # CSV format
│       ├── huggingface/                    # HuggingFace format
│       └── roberta_training/               # Training-ready format
│
├── dataset_tools/                  # Dataset processing utilities
│   ├── dataset.py                 # Download datasets
│   ├── dataset_cleaner.py         # Clean datasets
│   ├── dataset_combiner.py        # Combine datasets
│   ├── dataset_preprocessing.py   # Preprocess for training
│   ├── dataset_mapping.py         # Analyze dataset structures
│   └── validate_preprocessed_data.py # Validation tools
│
├── docs/                           # Documentation
│   ├── PREPROCESSING_SUMMARY.md   # Data preprocessing details
│   ├── PROJECT_STATUS.md          # Current project status
│   └── REORGANIZATION_SUMMARY.md  # Project structure changes
│
├── logs/                           # Training logs (git-ignored)
│   └── training_roberta_*.log     # Timestamped training logs
│
├── models/                         # Saved models (for distribution)
│
├── results/                        # Training results
│   ├── training_history_roberta.png        # Training curves
│   ├── training_history_roberta.json       # Metrics data
│   ├── test_metrics_roberta_loss.json      # Test metrics (best loss)
│   ├── test_metrics_roberta_f1.json        # Test metrics (best F1)
│   ├── classification_report_roberta_*.txt # Detailed reports
│   └── confusion_matrices_roberta_*.png    # Confusion matrices
│
└── scripts/                        # Utility scripts
    └── predict_emotion.py         # Emotion prediction script
```

## Changes Made

### New Structure
- **data/**: Centralized location for all datasets
- **docs/**: Documentation and project summaries
- **logs/**: Training and execution logs
- **models/**: For storing final production models
- **scripts/**: Reusable utility scripts

### Moved Files
- Documentation files → `docs/`
- Dataset files → `data/`
- Log files → `logs/`
- Created `predict_emotion.py` → `scripts/`

### Benefits
1. **Better Organization**: Clear separation of concerns
2. **Easier Navigation**: Find files by category
3. **Cleaner Root**: Less clutter in main directory
4. **Scalability**: Easy to add new components
5. **Standard Structure**: Follows common project patterns

## Path Updates Required

If you have scripts referencing old paths, update them:

**Old Path** → **New Path**
- `combined_dataset_clean.json` → `data/combined_dataset_clean.json`
- `preprocessed_data_roberta/` → `data/preprocessed_data_roberta/`
- `PREPROCESSING_SUMMARY.md` → `docs/PREPROCESSING_SUMMARY.md`
- Training logs → `logs/training_roberta_*.log`

## Training Script Updates

The main training script (`train_emotion_chatbot_roberta.py`) needs config updates:

```python
config = {
    'train_data': r"data\preprocessed_data_roberta\json\train.json",
    'val_data': r"data\preprocessed_data_roberta\json\validation.json",
    'test_data': r"data\preprocessed_data_roberta\json\test.json",
    # ... other configs
}
```
