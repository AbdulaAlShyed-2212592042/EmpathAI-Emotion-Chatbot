# EmpathAI-Emotion-Chatbot

A comprehensive emotion recognition and chatbot system built with RoBERTa transformer models. This project provides end-to-end solutions from dataset preprocessing to training production-ready emotion classification models.

## 🎯 Features

- **Production-Ready Emotion Model**: Fine-tuned RoBERTa-base model achieving 66.9% accuracy on 35-emotion classification
- **Multi-Dataset Processing**: 139K+ labeled examples from 5 different emotion datasets (GoEmotions, IMDB, Emotion, TweetEval, Yelp)
- **35 Emotion Categories**: Comprehensive emotion mapping across all datasets
- **GPU-Optimized Training**: Full CUDA support with TF32, gradient accumulation, and mixed precision
- **Advanced Training Features**: Early stopping, learning rate warmup, label smoothing, and F1-based model selection
- **Multiple Output Formats**: JSON, CSV, HuggingFace datasets, and training-ready formats
- **Comprehensive Checkpointing**: Saves best models by both loss and F1 score

## 🏆 Model Performance

**Best Model Metrics (Test Set - 20,897 examples):**
- **Exact Match Accuracy**: 66.9%
- **Hamming Accuracy**: 98.3%
- **F1 Score (Macro)**: 51.7%
- **F1 Score (Weighted)**: 70.3%
- **Precision (Macro)**: 59.0%
- **Recall (Macro)**: 48.7%

**Training Details:**
- Trained on **97,517 examples** (100% of training data, no cropping)
- 23 epochs completed (early stopping triggered)
- GPU: NVIDIA GeForce RTX 3060 (12GB VRAM)
- Training time: ~6 hours with full dataset

## 📊 Dataset Information

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

### Top Emotion Categories

**Most frequent emotions:**
- **positive**: 81.3% F1 | **negative**: 91.2% F1 | **neutral**: 77.6% F1
- **joy**: 86.8% F1 | **sadness**: 77.3% F1 | **anger**: 81.8% F1
- **admiration**: 67.6% F1 | **love**: 86.3% F1 | **excitement**: 83.2% F1

**Complete emotion set (35)**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral, positive, negative, plus Yelp ratings (1-5 stars)

## 🚀 Quick Start

### Installation

1. **Clone and setup**:
   ```bash
   git clone https://github.com/AbdulaAlShyed-2212592042/EmpathAI-Emotion-Chatbot.git
   cd EmpathAI-Emotion-Chatbot
   pip install -r requirements.txt
   ```

2. **Install PyTorch with GPU** (recommended):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU**:
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

### Train Model

```bash
python train_emotion_chatbot_roberta.py
```

**Training takes ~6 hours on GPU (RTX 3060) or ~20 hours on CPU.**

## 📦 Usage

### Predict Emotions (Inference)

Create `predict_emotion.py`:

```python
import torch
from transformers import RobertaTokenizer
from train_emotion_chatbot_roberta import RobertaEmotionChatbot

# Setup
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaEmotionChatbot(num_emotions=35, pretrained_model='roberta-base')

# Load checkpoint
checkpoint = torch.load('checkpoint/best_model_f1.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Predict
text = "I'm so happy and excited!"
encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
with torch.no_grad():
    emotion_logits, _ = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
    probs = torch.sigmoid(emotion_logits)[0]
    top_5 = torch.topk(probs, 5)
    print("Top emotions:", top_5)
```

Run: `python predict_emotion.py`

## 📁 Project Structure

```
EmpathAI-Emotion-Chatbot/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── train_emotion_chatbot_roberta.py   # Main training script
├── PREPROCESSING_SUMMARY.md            # Preprocessing documentation
├── PROJECT_STATUS.md                   # Project status
│
├── combined_dataset_clean.json         # Unified dataset (139K entries)
│
├── checkpoint/                         # Model checkpoints
│   ├── best_model_loss.pt             # Best model by validation loss
│   ├── best_model_f1.pt               # Best model by F1 score
│   ├── checkpoint_latest.pt           # Latest checkpoint
│   └── checkpoint_epoch_*.pt          # Periodic checkpoints
│
├── results/                            # Training results
│   ├── training_history_roberta.png   # Training curves
│   ├── test_metrics_roberta_*.json    # Test metrics
│   ├── classification_report_*.txt    # Detailed reports
│   └── confusion_matrices_*.png       # Confusion matrices
│
├── dataset_tools/                      # Dataset processing tools
│   ├── dataset.py                     # Download datasets
│   ├── dataset_mapping.py             # Analyze structures
│   ├── dataset_cleaner.py             # Clean datasets
│   ├── dataset_combiner.py            # Combine datasets
│   ├── dataset_preprocessing.py       # Preprocess for training
│   └── validate_preprocessed_data.py  # Validation tools
│
└── preprocessed_data_roberta/          # Preprocessed training data
    ├── json/                          # JSON format (train/val/test)
    ├── csv/                           # CSV format
    ├── huggingface/                   # HuggingFace datasets
    └── roberta_training/              # RoBERTa-specific files
```

## 🔬 Dataset Files

The `preprocessed_data_roberta/` folder contains training data in multiple formats:
- **JSON**: `json/train.json`, `json/validation.json`, `json/test.json`
- **CSV**: `csv/train.csv`, `csv/validation.csv`, `csv/test.csv`
- **HuggingFace**: `huggingface/` (ready for Transformers library)
- **RoBERTa Training**: `roberta_training/` (multi-emotion and binary sentiment)

## ⚙️ Hardware Requirements

**Minimum (CPU)**: 4+ cores, 16GB RAM, 5GB storage, ~20 hours training  
**Recommended (GPU)**: NVIDIA 8GB+ VRAM (RTX 3060/2060/3070/4060), ~6 hours training  
**Peak VRAM**: ~4-5GB during training

## 🔧 Troubleshooting

**CUDA Out of Memory**: Reduce `batch_size` from 16 to 8 in `train_emotion_chatbot_roberta.py`

**No GPU Detected**: Install CUDA PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Module Not Found**: Install dependencies:
```bash
pip install -r requirements.txt
```

## 🛠️ Development

### Customize Hyperparameters
Edit `train_emotion_chatbot_roberta.py`:
```python
config = {
    'batch_size': 16,           # Adjust for GPU memory
    'learning_rate': 2e-5,      # Lower for stability
    'num_epochs': 30,           # Max epochs
    'dropout': 0.3,             # Regularization
}
```

### Add New Datasets
Modify `LABEL_MAPPINGS` in `dataset_tools/dataset_mapping.py` and `dataset_tools/dataset_combiner.py`

### Future Enhancements
- 🔲 Web interface (Streamlit/Flask)
- 🔲 REST API for emotion detection
- 🔲 Model quantization for faster inference
- 🔲 Multi-lingual support

## 🤝 Contributing

Contributions welcome! Fork the repo, create a feature branch, commit changes, and open a Pull Request.

## 📄 License

Open source project. See repository for license details.

## 🙏 Acknowledgments

- **Datasets**: GoEmotions (Google), IMDB, Emotion, TweetEval, Yelp Reviews
- **Models**: Hugging Face Transformers (RoBERTa-base)
- **Framework**: PyTorch, Transformers, scikit-learn

## 📚 Citation

```bibtex
@misc{empathAI2025,
  title={EmpathAI: Emotion Recognition with RoBERTa Transformers},
  author={EmpathAI Team},
  year={2025},
  url={https://github.com/AbdulaAlShyed-2212592042/EmpathAI-Emotion-Chatbot}
}
```

---
**Built with ❤️ using RoBERTa, PyTorch, and 139K+ emotion-labeled examples**

