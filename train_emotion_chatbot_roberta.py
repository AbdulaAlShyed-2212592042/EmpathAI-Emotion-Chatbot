import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
from datetime import datetime
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score
)
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

# Configure logging with UTF-8 encoding to fix Unicode errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_roberta_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


class EmotionChatbotDataset(Dataset):
    """Dataset class for RoBERTa-based emotion chatbot."""
    
    def __init__(self, data_path: str, tokenizer: RobertaTokenizer, max_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON file with preprocessed data
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        self.emotion_to_idx = self._build_emotion_mapping()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Number of emotions: {len(self.emotion_to_idx)}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _build_emotion_mapping(self) -> Dict[str, int]:
        """Build emotion to index mapping."""
        emotions = set()
        for item in self.data:
            for emotion in item['emotion_names']:
                emotions.add(emotion)
        
        return {emotion: idx for idx, emotion in enumerate(sorted(emotions))}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from dataset."""
        item = self.data[idx]
        
        # Tokenize text using RoBERTa tokenizer
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get emotion labels (multi-hot encoding)
        emotion_labels = torch.zeros(len(self.emotion_to_idx))
        for emotion in item['emotion_names']:
            if emotion in self.emotion_to_idx:
                emotion_labels[self.emotion_to_idx[emotion]] = 1.0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'emotion_labels': emotion_labels,
            'text_str': item['text']
        }


class RobertaEmotionChatbot(nn.Module):
    """RoBERTa-based emotion chatbot model."""
    
    def __init__(self, num_emotions: int, dropout: float = 0.3, pretrained_model: str = 'roberta-base'):
        super().__init__()
        
        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        self.config = self.roberta.config
        hidden_size = self.config.hidden_size
        
        # Emotion classifier head
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotions)
        )
        
        # Response generation head (simplified for emotion-conditioned text generation)
        self.response_generator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + num_emotions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.config.vocab_size)
        )
        
        logger.info(f"Loaded pre-trained RoBERTa model: {pretrained_model}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                emotion_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            emotion_labels: Emotion labels for conditioning [batch, num_emotions]
        
        Returns:
            emotion_logits: Emotion predictions [batch, num_emotions]
            generation_logits: Generation predictions [batch, seq_len, vocab_size]
        """
        # Get RoBERTa embeddings
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation for emotion classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(cls_output)
        
        # For generation, use predicted emotions if not provided
        if emotion_labels is None:
            emotion_probs = torch.sigmoid(emotion_logits)
        else:
            emotion_probs = emotion_labels
        
        # Combine sequence output with emotion for generation
        batch_size, seq_len, hidden_size = outputs.last_hidden_state.shape
        emotion_expanded = emotion_probs.unsqueeze(1).expand(batch_size, seq_len, -1)
        combined = torch.cat([outputs.last_hidden_state, emotion_expanded], dim=-1)
        
        # Generate output logits
        generation_logits = self.response_generator(combined)
        
        return emotion_logits, generation_logits


class RobertaChatbotTrainer:
    """Trainer class for RoBERTa emotion chatbot."""
    
    def __init__(
        self,
        model: RobertaEmotionChatbot,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 2e-5,
        num_epochs: int = 30,
        model_dir: str = "roberta_models",
        results_dir: str = "results",
        checkpoint_dir: str = "checkpoint",
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 2
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimizer with weight decay for regularization
        self.optimizer = AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        self.warmup_steps = warmup_steps
        self.total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        
        # Loss functions with label smoothing for better generalization
        self.emotion_loss = nn.BCEWithLogitsLoss()
        self.generation_loss = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.max_patience = 7  # Increased patience for longer training
        self.current_step = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'emotion_loss': [],
            'generation_loss': [],
            'emotion_accuracy': []
        }
        
        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation and learning rate warmup."""
        self.model.train()
        total_loss = 0
        total_emotion_loss = 0
        total_generation_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            emotion_labels = batch['emotion_labels'].to(self.device)
            
            # Forward pass
            emotion_logits, generation_logits = self.model(
                input_ids, attention_mask, emotion_labels
            )
            
            # Compute losses
            emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
            
            # Generation loss (predict next token)
            gen_loss = self.generation_loss(
                generation_logits[:, :-1, :].reshape(-1, generation_logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )
            
            # Combined loss with gradient accumulation
            loss = (emotion_loss + 0.5 * gen_loss) / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Warmup learning rate
                if self.current_step < self.warmup_steps:
                    lr_scale = min(1.0, float(self.current_step + 1) / self.warmup_steps)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr_scale * 2e-5
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.current_step += 1
            
            # Update metrics (scale back the loss for display)
            actual_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += actual_loss
            total_emotion_loss += emotion_loss.item()
            total_generation_loss += gen_loss.item()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'emotion': f'{emotion_loss.item():.4f}',
                'gen': f'{gen_loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'emotion_loss': total_emotion_loss / len(self.train_loader),
            'generation_loss': total_generation_loss / len(self.train_loader)
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model with comprehensive metrics."""
        self.model.eval()
        total_loss = 0
        total_emotion_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                emotion_logits, generation_logits = self.model(
                    input_ids, attention_mask, emotion_labels
                )
                
                # Emotion loss
                emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
                
                # Generation loss
                gen_loss = self.generation_loss(
                    generation_logits[:, :-1, :].reshape(-1, generation_logits.size(-1)),
                    input_ids[:, 1:].reshape(-1)
                )
                
                loss = emotion_loss + 0.5 * gen_loss
                total_loss += loss.item()
                
                # Emotion accuracy
                emotion_pred = (torch.sigmoid(emotion_logits) > 0.5).float()
                correct = (emotion_pred == emotion_labels).all(dim=1).sum().item()
                total_emotion_correct += correct
                total_samples += input_ids.size(0)
                
                # Collect predictions for F1 score
                all_preds.append(emotion_pred.cpu().numpy())
                all_labels.append(emotion_labels.cpu().numpy())
        
        # Calculate F1 score
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'emotion_accuracy': total_emotion_correct / total_samples,
            'val_f1_macro': val_f1
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, is_best_f1: bool = False):
        """Save model checkpoint to checkpoint folder."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'current_step': self.current_step
        }
        
        # Save regular checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Always save latest checkpoint
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model based on validation loss
        if is_best:
            best_path = self.checkpoint_dir / 'best_model_loss.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"[BEST LOSS] Saved best model to {best_path}")
        
        # Save best model based on F1 score
        if is_best_f1:
            best_f1_path = self.checkpoint_dir / 'best_model_f1.pt'
            torch.save(checkpoint, best_f1_path)
            logger.info(f"[BEST F1] Saved best F1 model to {best_f1_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("="*80)
        logger.info("Starting RoBERTa Emotion Chatbot Training")
        logger.info("="*80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of epochs: {self.num_epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Emotion Loss: {train_metrics['emotion_loss']:.4f}")
            logger.info(f"  Generation Loss: {train_metrics['generation_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Emotion Accuracy: {val_metrics['emotion_accuracy']:.4f}")
            logger.info(f"Validation F1 (Macro): {val_metrics['val_f1_macro']:.4f}")
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_metrics['val_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # Save metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['emotion_loss'].append(train_metrics['emotion_loss'])
            self.history['generation_loss'].append(train_metrics['generation_loss'])
            self.history['emotion_accuracy'].append(val_metrics['emotion_accuracy'])
            if 'val_f1' not in self.history:
                self.history['val_f1'] = []
            self.history['val_f1'].append(val_metrics['val_f1_macro'])
            
            # Check for improvement (loss)
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                logger.info(f"*** New best validation loss: {val_metrics['val_loss']:.4f} (previous: {self.best_val_loss:.4f}) ***")
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Check for improvement (F1 score)
            is_best_f1 = val_metrics['val_f1_macro'] > self.best_val_f1
            if is_best_f1:
                logger.info(f"*** New best F1 score: {val_metrics['val_f1_macro']:.4f} (previous: {self.best_val_f1:.4f}) ***")
                self.best_val_f1 = val_metrics['val_f1_macro']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best, is_best_f1)
            
            # Plot history every 5 epochs
            if epoch % 5 == 0:
                self.plot_training_history()
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                logger.info(f"\n{'='*80}")
                logger.info(f"Early stopping triggered after {epoch} epochs")
                logger.info(f"No improvement in validation loss for {self.max_patience} consecutive epochs")
                logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                logger.info(f"Best F1 score: {self.best_val_f1:.4f}")
                logger.info(f"{'='*80}")
                break
        
        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss achieved: {self.best_val_loss:.4f}")
        logger.info(f"Best F1 score achieved: {self.best_val_f1:.4f}")
        self.plot_training_history()
        
        # Test with best loss model
        logger.info("\n" + "="*80)
        logger.info("Testing with BEST LOSS model...")
        logger.info("="*80)
        self.test(model_type='loss')
        
        # Test with best F1 model
        logger.info("\n" + "="*80)
        logger.info("Testing with BEST F1 model...")
        logger.info("="*80)
        self.test(model_type='f1')
    
    def test(self, model_type='loss'):
        """Test the model with comprehensive metrics."""
        logger.info("\n" + "="*80)
        logger.info(f"Testing on test set with best {model_type.upper()} model...")
        logger.info("="*80)
        
        # Load best model based on type
        if model_type == 'f1':
            best_model_path = self.checkpoint_dir / 'best_model_f1.pt'
        else:
            best_model_path = self.checkpoint_dir / 'best_model_loss.pt'
        
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")
        else:
            logger.warning(f"Best model not found at {best_model_path}, using current model")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                emotion_logits, _ = self.model(input_ids, attention_mask)
                emotion_probs = torch.sigmoid(emotion_logits)
                emotion_pred = (emotion_probs > 0.5).float()
                
                all_predictions.append(emotion_pred.cpu().numpy())
                all_labels.append(emotion_labels.cpu().numpy())
                all_probs.append(emotion_probs.cpu().numpy())
        
        # Concatenate all batches
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        all_probs = np.vstack(all_probs)
        
        # Compute metrics
        test_metrics = self._compute_test_metrics(all_predictions, all_labels, all_probs)
        
        # Save results with model type suffix
        suffix = f'_{model_type}' if model_type else ''
        self._save_test_results(test_metrics, all_predictions, all_labels, suffix)
        self._plot_confusion_matrices(all_predictions, all_labels, suffix)
        
        logger.info(f"\nTest Results Saved to: {self.results_dir}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1 Score (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"Test Precision (Macro): {test_metrics['precision_macro']:.4f}")
        logger.info(f"Test Recall (Macro): {test_metrics['recall_macro']:.4f}")
        
        return test_metrics
    
    def _compute_test_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                              probs: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive test metrics."""
        exact_match = (predictions == labels).all(axis=1).mean()
        hamming_acc = (predictions == labels).mean()
        
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
        
        recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
        per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)
        
        return {
            'accuracy': exact_match,
            'hamming_accuracy': hamming_acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_micro': precision_micro,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_micro': recall_micro,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'per_class_f1': per_class_f1.tolist(),
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist()
        }
    
    def _save_test_results(self, metrics: Dict, predictions: np.ndarray, labels: np.ndarray, suffix: str = ''):
        """Save test results to file."""
        metrics_file = self.results_dir / f'test_metrics_roberta{suffix}.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved test metrics to {metrics_file}")
        
        report_file = self.results_dir / f'classification_report_roberta{suffix}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ROBERTA EMOTION CHATBOT - TEST SET EVALUATION\n")
            f.write("="*80 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Exact Match Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Hamming Accuracy: {metrics['hamming_accuracy']:.4f}\n\n")
            
            f.write("F1 Scores:\n")
            f.write(f"  Micro:    {metrics['f1_micro']:.4f}\n")
            f.write(f"  Macro:    {metrics['f1_macro']:.4f}\n")
            f.write(f"  Weighted: {metrics['f1_weighted']:.4f}\n\n")
            
            f.write("Precision:\n")
            f.write(f"  Micro:    {metrics['precision_micro']:.4f}\n")
            f.write(f"  Macro:    {metrics['precision_macro']:.4f}\n")
            f.write(f"  Weighted: {metrics['precision_weighted']:.4f}\n\n")
            
            f.write("Recall:\n")
            f.write(f"  Micro:    {metrics['recall_micro']:.4f}\n")
            f.write(f"  Macro:    {metrics['recall_macro']:.4f}\n")
            f.write(f"  Weighted: {metrics['recall_weighted']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Emotion':<20} {'F1':>10} {'Precision':>12} {'Recall':>10}\n")
            f.write("-" * 80 + "\n")
            
            emotion_names = sorted(self.train_loader.dataset.emotion_to_idx.keys())
            for i, emotion in enumerate(emotion_names):
                f1 = metrics['per_class_f1'][i]
                prec = metrics['per_class_precision'][i]
                rec = metrics['per_class_recall'][i]
                f.write(f"{emotion:<20} {f1:>10.4f} {prec:>12.4f} {rec:>10.4f}\n")
        
        logger.info(f"Saved classification report to {report_file}")
    
    def _plot_confusion_matrices(self, predictions: np.ndarray, labels: np.ndarray, suffix: str = ''):
        """Plot and save confusion matrices for top emotions."""
        emotion_to_idx = self.train_loader.dataset.emotion_to_idx
        emotion_names = sorted(emotion_to_idx.keys())
        
        emotion_counts = labels.sum(axis=0)
        top_emotion_indices = np.argsort(emotion_counts)[-10:][::-1]
        
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes = axes.flatten()
        
        for i, emotion_idx in enumerate(top_emotion_indices):
            cm = confusion_matrix(labels[:, emotion_idx], predictions[:, emotion_idx])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            axes[i].set_title(f'{emotion_names[emotion_idx]}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        confusion_file = self.results_dir / f'confusion_matrices_roberta{suffix}.png'
        plt.savefig(confusion_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrices to {confusion_file}")
    
    def plot_training_history(self):
        """Plot and save training history graphs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training and Validation Loss
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss', linewidth=2, marker='o')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('RoBERTa: Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Emotion Accuracy and F1
        axes[0, 1].plot(self.history['emotion_accuracy'], label='Validation Accuracy', 
                       color='green', linewidth=2, marker='D')
        if 'val_f1' in self.history:
            axes[0, 1].plot(self.history['val_f1'], label='Validation F1 (Macro)', 
                           color='orange', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontsize=12)
        axes[0, 1].set_title('Emotion Metrics', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Component Losses
        axes[1, 0].plot(self.history['emotion_loss'], label='Emotion Loss', linewidth=2, marker='^')
        axes[1, 0].plot(self.history['generation_loss'], label='Generation Loss', linewidth=2, marker='v')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Component Losses', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning Curve
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[1, 1].plot(epochs, self.history['train_loss'], 'o-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_loss'], 's-', label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss', fontsize=12)
        axes[1, 1].set_title('Learning Curve', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        history_file = self.results_dir / 'training_history_roberta.png'
        plt.savefig(history_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training history plots to {history_file}")
        
        history_json = self.results_dir / 'training_history_roberta.json'
        with open(history_json, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history data to {history_json}")


def main():
    """Main training function."""
    # Configuration with optimized hyperparameters
    config = {
        'train_data': r"C:\Users\sslue\AI chatbot\EmpathAI-Emotion-Chatbot\preprocessed_data_roberta\json\train.json",
        'val_data': r"C:\Users\sslue\AI chatbot\EmpathAI-Emotion-Chatbot\preprocessed_data_roberta\json\validation.json",
        'test_data': r"C:\Users\sslue\AI chatbot\EmpathAI-Emotion-Chatbot\preprocessed_data_roberta\json\test.json",
        'batch_size': 16,  # Batch size for GPU memory efficiency
        'max_length': 128,  # Max sequence length
        'dropout': 0.3,  # Dropout for regularization
        'learning_rate': 2e-5,  # Initial learning rate with warmup
        'num_epochs': 30,  # Increased epochs with early stopping
        'pretrained_model': 'roberta-base',
        'model_dir': 'roberta_emotion_models',
        'results_dir': 'results',
        'checkpoint_dir': 'checkpoint',  # Dedicated checkpoint folder
        'warmup_steps': 500,  # Learning rate warmup steps
        'gradient_accumulation_steps': 2  # Gradient accumulation for stability
    }
    
    # Device - AUTOMATICALLY USES GPU IF AVAILABLE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("="*80)
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"[GPU] GPU DETECTED: {torch.cuda.get_device_name(0)}")
        logger.info(f"[GPU] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"[GPU] CUDA Version: {torch.version.cuda}")
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"[GPU] TF32 enabled for improved performance")
    else:
        logger.warning("[WARNING] No GPU detected - Training will be slow on CPU")
        logger.warning("[WARNING] Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    logger.info("="*80)
    
    # Initialize tokenizer
    logger.info(f"Loading RoBERTa tokenizer: {config['pretrained_model']}")
    tokenizer = RobertaTokenizer.from_pretrained(config['pretrained_model'])
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = EmotionChatbotDataset(config['train_data'], tokenizer, config['max_length'])
    val_dataset = EmotionChatbotDataset(config['val_data'], tokenizer, config['max_length'])
    test_dataset = EmotionChatbotDataset(config['test_data'], tokenizer, config['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Create model
    logger.info("Creating RoBERTa emotion chatbot model...")
    model = RobertaEmotionChatbot(
        num_emotions=len(train_dataset.emotion_to_idx),
        dropout=config['dropout'],
        pretrained_model=config['pretrained_model']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer with enhanced features
    trainer = RobertaChatbotTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        model_dir=config['model_dir'],
        results_dir=config['results_dir'],
        checkpoint_dir=config['checkpoint_dir'],
        warmup_steps=config['warmup_steps'],
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    
    # Train
    trainer.train()
    
    logger.info("\n" + "="*80)
    logger.info("RoBERTa Training Pipeline Completed Successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
