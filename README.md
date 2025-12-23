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

---

> **📖 New to the project?** Check out the [Quick Start Guide](docs/QUICK_START_GUIDE.md) for a 5-minute setup!

---

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

### Use Pre-trained Model

```bash
# Run prediction script
python scripts/predict_emotion.py
```

### Train Model

```bash
python train_emotion_chatbot_roberta.py
```

**Training takes ~6 hours on GPU (RTX 3060) or ~20 hours on CPU.**

## 📁 Project Structure

```
EmpathAI-Emotion-Chatbot/
│
├── data/                          # Dataset files
│   ├── preprocessed_data_roberta/ # Preprocessed training data
│   │   ├── json/                  # Train/test/val splits
│   │   ├── csv/                   # CSV format datasets
│   │   └── huggingface/           # HuggingFace Dataset format
│   └── combined_dataset_clean.json
│
├── models/                        # Trained model files
│   ├── best_model_loss.pt         # Best model by validation loss
│   └── best_model_f1.pt           # Best model by F1 score
│
├── checkpoint/                    # Training checkpoints
│   ├── best_model_loss.pt
│   ├── best_model_f1.pt
│   └── checkpoint_latest.pt
│
├── results/                       # Training results
│   ├── test_metrics.json
│   ├── training_history.json
│   └── confusion_matrix_*.png
│
├── logs/                          # Training logs
│   └── training_roberta_*.log
│
├── scripts/                       # Utility scripts
│   └── predict_emotion.py         # Inference script
│
├── docs/                          # Documentation
│   ├── PREPROCESSING_SUMMARY.md
│   ├── PROJECT_STATUS.md
│   └── PROJECT_ORGANIZATION.md
│
├── dataset_tools/                 # Data processing tools
│   ├── dataset_preprocessing.py
│   ├── dataset_cleaner.py
│   └── dataset_combiner.py
│
├── train_emotion_chatbot_roberta.py  # Main training script
├── requirements.txt
└── README.md
```

## 📊 Results & Achievements

### Model Performance Summary

Our fine-tuned RoBERTa-base model demonstrates strong performance across 35 emotion categories:

| Metric | Score |
|--------|-------|
| **Exact Match Accuracy** | 66.9% |
| **Hamming Accuracy** | 98.3% |
| **F1 Score (Macro)** | 51.7% |
| **F1 Score (Weighted)** | 70.3% |
| **Precision (Macro)** | 59.0% |
| **Recall (Macro)** | 48.7% |

### Top Performing Emotions

| Emotion | F1 Score | Samples |
|---------|----------|---------|
| **negative** | 91.2% | 25,000 |
| **positive** | 81.3% | 25,000 |
| **joy** | 86.8% | 9,709 |
| **love** | 86.3% | 4,217 |
| **excitement** | 83.2% | 2,516 |
| **anger** | 81.8% | 6,787 |
| **neutral** | 77.6% | 17,770 |
| **sadness** | 77.3% | 8,748 |

### Training Efficiency

- **Total Training Time**: ~6 hours on NVIDIA RTX 3060 (12GB VRAM)
- **Epochs Completed**: 23 (early stopping triggered)
- **Training Samples**: 97,517 (100% of available data)
- **Validation Samples**: 20,897
- **Test Samples**: 20,897
- **Model Parameters**: 164M (RoBERTa-base)
- **Peak GPU Memory**: ~4-5GB VRAM

### Key Features Implemented

✅ **Advanced Training Techniques**:
- Learning rate warmup (500 steps)
- Gradient accumulation (effective batch size: 32)
- Label smoothing (0.1)
- Early stopping (patience: 7 epochs)
- Dual model selection (best loss & best F1)

✅ **Multi-Dataset Integration**:
- 5 diverse emotion datasets combined
- 139,311 preprocessed samples
- 35 unified emotion labels
- Multi-label and single-label support

## 📦 Usage

### Predict Emotions (Inference)

Use the pre-built inference script:

```bash
# Interactive mode
python scripts/predict_emotion.py

# Batch prediction
python scripts/predict_emotion.py --texts "I'm so happy!" "This is terrible."
```

Or create your own prediction script:

```python
import torch
from transformers import RobertaTokenizer
from train_emotion_chatbot_roberta import RobertaEmotionChatbot

# Setup
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaEmotionChatbot(num_emotions=35, pretrained_model='roberta-base')

# Load checkpoint
checkpoint = torch.load('models/best_model_f1.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Predict
text = "I'm so happy and excited!"
encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
with a torch.no_grad():
    emotion_logits, _ = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
    probs = torch.sigmoid(emotion_logits)[0]
    top_5 = torch.topk(probs, 5)
    print("Top emotions:", top_5)
```

### Train Custom Model

```bash
# Full training (30 epochs with early stopping)
python train_emotion_chatbot_roberta.py

# Monitor training
tail -f logs/training_roberta_*.log

# Check results
cat results/test_metrics.json
```

**Training Configuration:**
- Batch size: 16 (GPU) or 8 (CPU)
- Learning rate: 2e-5 with warmup (500 steps)
- Max epochs: 30 with early stopping (patience: 7)
- Gradient accumulation: 2 steps
- Label smoothing: 0.1

## 🔬 Dataset Files

The `data/preprocessed_data_roberta/` folder contains training data in multiple formats:
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

## 👥 Authors

**Md Abdula Al Shyed**
- Student ID: 2212592042
- Email: abdula.shyed@northsouth.edu
- GitHub: [@AbdulaAlShyed-2212592042](https://github.com/AbdulaAlShyed-2212592042)

**Md. Raduain Hossain Rimon**
- Student ID: 2021995642
- Email: raduain.rimon@northsouth.edu

*North South University*

## 📄 License

Open source project. See the repository for license details.

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
**Built with  using RoBERTa, PyTorch, and 139K+ emotion-labeled examples**

