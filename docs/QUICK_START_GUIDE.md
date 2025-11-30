# EmpathAI Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Installation

```bash
# Clone repository
git clone https://github.com/AbdulaAlShyed-2212592042/EmpathAI-Emotion-Chatbot.git
cd EmpathAI-Emotion-Chatbot

# Install dependencies
pip install -r requirements.txt

# Install GPU support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify Setup

```bash
# Check GPU
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Expected output: CUDA Available: True (if GPU present)
```

### 3. Use Pre-trained Model

```bash
# Run interactive prediction
python scripts/predict_emotion.py

# Or predict specific texts
python scripts/predict_emotion.py --texts "I'm so happy!" "This is terrible"
```

### 4. Train Your Own Model

```bash
# Start training (30 epochs with early stopping)
python train_emotion_chatbot_roberta.py

# Training takes ~6 hours on RTX 3060 GPU or ~20 hours on CPU
```

### 5. Monitor Training

```bash
# View live logs
tail -f logs/training_roberta_*.log

# Check results after training
cat results/test_metrics.json
```

## 📊 What to Expect

### Model Performance
- **Accuracy**: ~67% on 35-emotion classification
- **F1 Score**: ~70% (weighted), ~52% (macro)
- **Training Time**: 6 hours (GPU) / 20 hours (CPU)

### Output Files
After training, you'll find:
- `models/best_model_f1.pt` - Best model by F1 score
- `models/best_model_loss.pt` - Best model by validation loss
- `checkpoint/` - Training checkpoints
- `results/` - Metrics, confusion matrices, training history
- `logs/` - Detailed training logs

## 🎯 Common Use Cases

### Predict Single Text
```python
from scripts.predict_emotion import load_model, predict_emotions

model, tokenizer, device = load_model()
results = predict_emotions(["I love this!"], model, tokenizer, device)
print(results)
```

### Batch Prediction
```python
texts = [
    "I'm so happy and excited!",
    "This makes me angry",
    "I feel sad and disappointed"
]
results = predict_emotions(texts, model, tokenizer, device)
for text, emotions in zip(texts, results):
    print(f"\n{text}")
    for emotion, score in emotions:
        print(f"  {emotion}: {score:.1%}")
```

### Custom Training
Edit `train_emotion_chatbot_roberta.py` config:
```python
config = {
    'batch_size': 16,  # Reduce to 8 if out of memory
    'learning_rate': 2e-5,
    'num_epochs': 30,
    'early_stopping_patience': 7,
    # ... other settings
}
```

## 🔧 Troubleshooting

### GPU Not Detected
```bash
# Reinstall CUDA PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Error
Reduce batch size in `train_emotion_chatbot_roberta.py`:
```python
config = {
    'batch_size': 8,  # Changed from 16
    # ...
}
```

### Training Too Slow
- Enable GPU if available
- Reduce dataset size for quick testing
- Use fewer epochs for initial experiments

## 📁 Important Files

| File/Folder | Purpose |
|-------------|---------|
| `train_emotion_chatbot_roberta.py` | Main training script |
| `scripts/predict_emotion.py` | Inference script |
| `data/preprocessed_data_roberta/` | Training data |
| `models/` | Trained model files |
| `checkpoint/` | Training checkpoints |
| `results/` | Metrics and visualizations |
| `logs/` | Training logs |

## 💡 Tips

1. **First Time Users**: Start with the prediction script to test pre-trained models
2. **Training**: Use GPU for faster training (6 hours vs 20 hours)
3. **Monitoring**: Keep an eye on `logs/training_roberta_*.log` during training
4. **Checkpoints**: Both loss-based and F1-based models are saved automatically
5. **Early Stopping**: Training stops automatically if no improvement for 7 epochs

## 🎓 For Students/Researchers

### Dataset Information
- 139,311 preprocessed samples
- 5 emotion datasets combined (GoEmotions, IMDB, Emotion, TweetEval, Yelp)
- 35 unified emotion labels
- 70/15/15 train/validation/test split

### Model Architecture
- Base: RoBERTa-base (164M parameters)
- Fine-tuning: Multi-label emotion classification
- Techniques: Label smoothing, warmup, gradient accumulation, early stopping

### Results Included
- Test metrics (JSON)
- Confusion matrices (PNG)
- Training history (JSON)
- Classification reports (TXT)

---

**Need More Help?** Check the main [README.md](../README.md) or documentation in `docs/`
