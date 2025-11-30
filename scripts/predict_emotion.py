"""
Emotion Prediction Script
Predict emotions from text using the trained RoBERTa model.
"""

import torch
from transformers import RobertaTokenizer
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_emotion_chatbot_roberta import RobertaEmotionChatbot


# Emotion labels (in order)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral', 'positive',
    'negative', '1 star', '2 stars', '3 stars', '4 stars', '5 stars'
]


def load_model(checkpoint_path='models/best_model_f1.pt', device=None):
    """Load the trained emotion detection model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Load model
    model = RobertaEmotionChatbot(num_emotions=35, pretrained_model='roberta-base')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def predict_emotions(text, model, tokenizer, device, top_k=5):
    """Predict emotions from text."""
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        emotion_logits, _ = model(input_ids, attention_mask)
        emotion_probs = torch.sigmoid(emotion_logits)[0]
    
    # Get top emotions
    top_indices = torch.topk(emotion_probs, k=top_k).indices.cpu().numpy()
    top_probs = torch.topk(emotion_probs, k=top_k).values.cpu().numpy()
    
    results = []
    for idx, prob in zip(top_indices, top_probs):
        results.append((EMOTION_LABELS[idx], float(prob)))
    
    return results


def main():
    """Main prediction function."""
    print("Loading model...")
    model, tokenizer, device = load_model()
    print(f"Model loaded on {device}")
    print("-" * 60)
    
    # Example texts
    texts = [
        "I'm so happy and excited about this amazing news!",
        "This movie was terrible and I'm very disappointed.",
        "I'm not sure what to think about this situation.",
        "Thank you so much for your help, I really appreciate it!",
    ]
    
    for text in texts:
        print(f"\nText: {text}")
        emotions = predict_emotions(text, model, tokenizer, device, top_k=3)
        print("Top 3 emotions:")
        for emotion, prob in emotions:
            bar = "█" * int(prob * 30)
            print(f"  {emotion:15s} {bar} {prob:.2%}")
    
    print("\n" + "=" * 60)
    print("Interactive Mode - Enter your own text")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        text = input("\nEnter text: ").strip()
        if text.lower() == 'quit':
            break
        if not text:
            continue
        
        emotions = predict_emotions(text, model, tokenizer, device, top_k=5)
        print("\nDetected emotions:")
        for emotion, prob in emotions:
            bar = "█" * int(prob * 30)
            print(f"  {emotion:15s} {bar} {prob:.2%}")


if __name__ == "__main__":
    main()
