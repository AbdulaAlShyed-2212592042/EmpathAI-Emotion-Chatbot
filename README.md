# EmpathAI-Emotion-Chatbot

A comprehensive dataset preprocessing pipeline for emotion recognition and sentiment analysis. This project provides tools to download, clean, combine, and preprocess multiple emotion datasets for training transformer models like RoBERTa.

## Features

- **Multi-Dataset Processing**: Works with 139K+ labeled examples from 5 different emotion datasets
- **35 Emotion Categories**: Unified emotion mapping across all datasets (GoEmotions, IMDB, Emotion, TweetEval, Yelp)
- **RoBERTa-Ready Preprocessing**: Optimized text cleaning and tokenization for transformer models
- **Multiple Output Formats**: JSON, CSV, HuggingFace datasets, and training-ready formats
- **Comprehensive Dataset Tools**: Download, clean, map, combine, and preprocess emotion datasets

## Dataset Information

This project processes and combines multiple emotion datasets into a unified format suitable for training transformer models:

| Dataset | Entries | Emotions | Type | Use Case |
|---------|---------|----------|------|----------|
| **GoEmotions** | 54,263 | 28 emotions | Multi-label | Fine-grained emotion detection |
| **IMDB** | 50,000 | positive/negative | Sentiment | Movie review sentiment |
| **Emotion** | 20,000 | 6 basic emotions | Single-label | Basic emotion classification |
| **Yelp Reviews** | 10,000 | 1-5 stars | Rating | Review sentiment analysis |
| **TweetEval** | 5,052 | 4 emotions | Single-label | Social media emotion patterns |
| **Total** | **139,315** | Various | Mixed | Comprehensive emotion training |

### Processed Dataset Features

- **139,311 preprocessed entries** (4 entries removed due to quality issues)
- **35 unified emotion labels** with proper mapping
- **Train/Validation/Test splits**: 70%/15%/15% (97,517/20,897/20,897)
- **RoBERTa-optimized**: Proper tokenization, ~118 avg tokens, 5% truncation rate
- **Multiple training formats**: Multi-emotion classification and binary sentiment

### Emotion Categories (35 Total)

**Top emotions by frequency:**
- **positive/negative**: 25,000 each (IMDB sentiment)
- **neutral**: 17,770 (GoEmotions)
- **joy**: 9,709 | **sadness**: 8,748 | **anger**: 6,787
- **admiration**: 5,122 | **love**: 4,217 | **approval**: 3,687

**Complete emotion set**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral, positive, negative, plus Yelp star ratings (1-5 stars)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AbdulaAlShyed-2212592042/EmpathAI-Emotion-Chatbot.git
   cd EmpathAI-Emotion-Chatbot
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start - Complete Pipeline

**Run the full preprocessing pipeline**:
```bash
# Download and combine all datasets, then preprocess for RoBERTa
python dataset_tools/dataset.py --all
python dataset_tools/dataset_cleaner.py
python dataset_tools/dataset_combiner.py
python dataset_tools/dataset_preprocessing.py --input combined_dataset_clean.json --output preprocessed_data_roberta
```

**Validate the results**:
```bash
python dataset_tools/validate_preprocessed_data.py
```

### Step-by-Step Dataset Processing

**1. Download datasets** from Hugging Face:
```bash
# Download all default datasets
python dataset_tools/dataset.py --all

# Download specific datasets
python dataset_tools/dataset.py --datasets "go_emotions,emotion"

# Download with row limits (for testing)
python dataset_tools/dataset.py --datasets "imdb" --limit 1000
```

**2. Analyze dataset structure**:
```bash
# Show structure and samples from all datasets
python dataset_tools/dataset_mapping.py
```

**3. Clean datasets** (remove unlabeled entries):
```bash
# Clean and save to dataset_cleaned/
python dataset_tools/dataset_cleaner.py
```

**4. Combine datasets** into single JSON:
```bash
# Create unified dataset with emotion labels
python dataset_tools/dataset_combiner.py
```

**5. Preprocess for training**:
```bash
# Create RoBERTa-ready training data
python dataset_tools/dataset_preprocessing.py --input combined_dataset_clean.json --output preprocessed_data_roberta
```

### Using Preprocessed Data

**Load HuggingFace format**:
```python
from datasets import load_from_disk
dataset = load_from_disk('preprocessed_data_roberta/huggingface')
train_dataset = dataset['train']
```

**Load JSON format**:
```python
import json
with open('preprocessed_data_roberta/roberta_training/train_binary_sentiment.json', 'r') as f:
    train_data = json.load(f)
```

**Load CSV format**:
```python
import pandas as pd
df = pd.read_csv('preprocessed_data_roberta/csv/train.csv')
```

## Project Structure

```
EmpathAI-Emotion-Chatbot/
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── PREPROCESSING_SUMMARY.md        # Detailed preprocessing documentation
├── PROJECT_STATUS.md               # Current project status
│
├── combined_dataset_clean.json     # Main dataset (139K emotion entries)
│
├── dataset_tools/                  # Dataset processing tools
│   ├── dataset.py                  # Download datasets from Hugging Face
│   ├── dataset_mapping.py          # Analyze and map dataset structures
│   ├── dataset_cleaner.py          # Remove unlabeled/invalid entries
│   ├── dataset_combiner.py         # Combine datasets into single JSON
│   ├── dataset_preprocessing.py    # Preprocess data for RoBERTa training
│   ├── validate_preprocessed_data.py # Validate and demonstrate preprocessed data
│   ├── train_roberta_template.py   # Template script for RoBERTa training
│   └── README.md                   # Dataset tools documentation
│
└── preprocessed_data_roberta/      # Preprocessed training data
    ├── json/                       # JSON format (train/val/test)
    ├── csv/                        # CSV format (train/val/test)
    ├── huggingface/                # HuggingFace dataset format
    ├── roberta_training/           # RoBERTa-specific training files
    └── metadata.json               # Complete preprocessing metadata
```

## Preprocessed Dataset Files

### Available Formats

1. **HuggingFace Dataset Format** (`preprocessed_data_roberta/huggingface/`)
   - Train: 97,517 examples | Validation: 20,897 | Test: 20,897
   - Features: `text`, `labels`, `emotion_names`, `dataset_source`
   - Ready for direct use with Transformers library

2. **JSON Format** (`preprocessed_data_roberta/json/`)
   - Complete feature set including tokenization info
   - Fields: `text`, `emotion_ids`, `emotion_names`, `dataset_source`, `is_multilabel`, etc.

3. **CSV Format** (`preprocessed_data_roberta/csv/`)
   - Flat format suitable for pandas analysis
   - All preprocessing statistics included

4. **RoBERTa Training Format** (`preprocessed_data_roberta/roberta_training/`)
   - `*_multi_emotion.json`: 35-class emotion classification
   - `*_binary_sentiment.json`: 3-class sentiment (positive/negative/neutral)
   - `label_mappings.json`: Complete label mapping information

### Dataset Statistics

- **Text Length**: Average 563 characters, 102 words, 118 RoBERTa tokens
- **Truncation Rate**: 5% (texts longer than 512 tokens)
- **Multilabel Rate**: 6.3% (entries with multiple emotions)
- **Quality**: No missing values, UTF-8 encoded, validated data

## Training Your Own Models

### Using the Template Script

```bash
# Modify train_roberta_template.py for your needs
python dataset_tools/train_roberta_template.py
```

### Recommended Hyperparameters

- **Model**: `roberta-base` (or `roberta-large` for better performance)
- **Learning Rate**: 2e-5
- **Batch Size**: 16-32 (adjust based on GPU memory)
- **Epochs**: 3-5
- **Max Length**: 512 tokens
- **Hardware**: GPU with 8GB+ VRAM recommended

### Training Options

1. **Multi-Emotion Classification** (35 classes)
   - Use `preprocessed_data_roberta/roberta_training/train_multi_emotion.json`
   - Multi-label classification task
   - Good for fine-grained emotion detection

2. **Binary Sentiment Classification** (3 classes)
   - Use `preprocessed_data_roberta/roberta_training/train_binary_sentiment.json`
   - Simpler task, faster training
   - Good for general sentiment analysis

## Development and Customization

### Adding New Datasets
Modify `LABEL_MAPPINGS` in `dataset_tools/dataset_mapping.py` and `dataset_tools/dataset_combiner.py` to add new emotion datasets.

### Custom Preprocessing
Edit `dataset_tools/dataset_preprocessing.py` to customize:
- Text cleaning rules
- Tokenization parameters
- Label mapping strategies
- Output formats

### Extending the Pipeline
The modular design makes it easy to:
- Add new preprocessing steps
- Support different model architectures
- Create custom training formats
- Integrate with different ML frameworks

## Future Development

This project currently focuses on dataset processing and preparation. Future enhancements may include:

- **Emotion Detection Models**: Fine-tuned RoBERTa/BERT models for emotion classification
- **Conversational AI Integration**: ChatGPT/LLM integration with emotion awareness
- **Web Interface**: Streamlit or Flask app for interactive emotion analysis
- **API Service**: REST API for emotion detection and sentiment analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source. See the repository for license details.

## Acknowledgments

- **Datasets**: GoEmotions (Google), IMDB, Emotion, TweetEval, Yelp Reviews
- **Models**: Hugging Face Transformers (RoBERTa)
- **Framework**: PyTorch, Transformers, Datasets, scikit-learn

## Citation

If you use this dataset preprocessing pipeline in your research, please cite:

```bibtex
@misc{empathAI2025,
  title={EmpathAI: Emotion Dataset Preprocessing Pipeline},
  author={EmpathAI Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/AbdulaAlShyed-2212592042/EmpathAI-Emotion-Chatbot}
}
```

