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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class EmotionChatbotDataset(Dataset):
    """Dataset class for emotion-aware chatbot training."""
    
    def __init__(self, data_path: str, vocab: Optional[Dict] = None, max_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON file with preprocessed data
            vocab: Vocabulary dictionary (if None, will be built)
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.emotion_to_idx = self._build_emotion_mapping()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        logger.info(f"Number of emotions: {len(self.emotion_to_idx)}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from data."""
        vocab = {
            '<PAD>': 0,
            '<SOS>': 1,  # Start of sequence
            '<EOS>': 2,  # End of sequence
            '<UNK>': 3   # Unknown token
        }
        
        word_freq = defaultdict(int)
        for item in self.data:
            words = item['text'].lower().split()
            for word in words:
                word_freq[word] += 1
        
        # Add words that appear at least twice
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if freq >= 2 and len(vocab) < 50000:  # Limit vocab size
                vocab[word] = len(vocab)
        
        return vocab
    
    def _build_emotion_mapping(self) -> Dict[str, int]:
        """Build emotion to index mapping."""
        emotions = set()
        for item in self.data:
            for emotion in item['emotion_names']:
                emotions.add(emotion)
        
        return {emotion: idx for idx, emotion in enumerate(sorted(emotions))}
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices."""
        words = text.lower().split()
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Truncate or pad
        if len(sequence) > self.max_length - 2:
            sequence = sequence[:self.max_length - 2]
        
        # Add SOS and EOS tokens
        sequence = [self.vocab['<SOS>']] + sequence + [self.vocab['<EOS>']]
        
        # Pad to max length
        while len(sequence) < self.max_length:
            sequence.append(self.vocab['<PAD>'])
        
        return sequence
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from dataset."""
        item = self.data[idx]
        
        # Convert text to sequence
        text_seq = self.text_to_sequence(item['text'])
        
        # Get emotion labels (multi-hot encoding)
        emotion_labels = torch.zeros(len(self.emotion_to_idx))
        for emotion in item['emotion_names']:
            if emotion in self.emotion_to_idx:
                emotion_labels[self.emotion_to_idx[emotion]] = 1.0
        
        # Get primary emotion
        primary_emotion_idx = self.emotion_to_idx.get(
            item['emotion_names'][0] if item['emotion_names'] else 'neutral',
            0
        )
        
        return {
            'text': torch.LongTensor(text_seq),
            'emotion_labels': emotion_labels,
            'primary_emotion': torch.LongTensor([primary_emotion_idx]),
            'text_str': item['text']
        }


class EmotionAttention(nn.Module):
    """Emotion-aware attention mechanism."""
    
    def __init__(self, hidden_size: int, emotion_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.emotion_size = emotion_size
        
        self.attention = nn.Linear(hidden_size + emotion_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                emotion_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute emotion-aware attention.
        
        Args:
            hidden: Decoder hidden state [batch, hidden_size]
            encoder_outputs: Encoder outputs [batch, seq_len, hidden_size]
            emotion_vector: Emotion embedding [batch, emotion_size]
        
        Returns:
            context: Context vector [batch, hidden_size]
            attention_weights: Attention weights [batch, seq_len]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Expand hidden and emotion to match encoder outputs
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        emotion_expanded = emotion_vector.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate with emotion information
        combined = torch.cat([hidden_expanded, emotion_expanded], dim=2)
        
        # Compute attention scores
        energy = torch.tanh(self.attention(combined))
        attention_weights = F.softmax(self.v(energy).squeeze(2), dim=1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights


class GRUEncoder(nn.Module):
    """GRU-based encoder for emotion recognition."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input sequences [batch, seq_len]
        
        Returns:
            outputs: Encoder outputs [batch, seq_len, hidden_size*2]
            hidden: Final hidden state [num_layers*2, batch, hidden_size]
        """
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class GRUDecoder(nn.Module):
    """GRU-based decoder with emotion-aware attention."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 emotion_size: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.emotion_embedding = nn.Linear(emotion_size, emotion_size)
        self.attention = EmotionAttention(hidden_size * 2, emotion_size)
        
        self.gru = nn.GRU(
            embedding_dim + hidden_size * 2 + emotion_size,
            hidden_size * 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor, 
                encoder_outputs: torch.Tensor, emotion_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through decoder.
        
        Args:
            x: Input token [batch, 1]
            hidden: Previous hidden state [num_layers, batch, hidden_size*2]
            encoder_outputs: Encoder outputs [batch, seq_len, hidden_size*2]
            emotion_vector: Emotion embedding [batch, emotion_size]
        
        Returns:
            output: Output predictions [batch, vocab_size]
            hidden: Updated hidden state
            attention_weights: Attention weights
        """
        x = x.unsqueeze(1) if x.dim() == 1 else x
        
        embedded = self.dropout(self.embedding(x))
        
        # Emotion-aware attention
        emotion_emb = F.relu(self.emotion_embedding(emotion_vector))
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs, emotion_emb
        )
        
        # Concatenate embedding, context, and emotion
        emotion_expanded = emotion_emb.unsqueeze(1)
        context_expanded = context.unsqueeze(1)
        
        rnn_input = torch.cat([embedded, context_expanded, emotion_expanded], dim=2)
        
        output, hidden = self.gru(rnn_input, hidden)
        output = self.fc(output.squeeze(1))
        
        return output, hidden, attention_weights


class EmotionChatbotModel(nn.Module):
    """Complete emotion-aware chatbot model."""
    
    def __init__(self, vocab_size: int, num_emotions: int, embedding_dim: int = 256,
                 hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.encoder = GRUEncoder(vocab_size, embedding_dim, hidden_size, num_layers, dropout)
        self.decoder = GRUDecoder(vocab_size, embedding_dim, hidden_size, num_emotions, num_layers, dropout)
        
        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_emotions)
        )
        
        # Bridge between encoder and decoder (handles bidirectional to unidirectional)
        # Encoder hidden: [num_layers*2, batch, hidden_size]
        # Decoder needs: [num_layers, batch, hidden_size*2]
        self.num_layers = num_layers
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            src: Source sequences [batch, seq_len]
            tgt: Target sequences [batch, seq_len] (optional)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Decoder outputs [batch, seq_len, vocab_size]
            emotion_logits: Emotion predictions [batch, num_emotions]
            attention_weights: Attention weights
        """
        batch_size = src.size(0)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Emotion classification from encoder output
        pooled = encoder_outputs.mean(dim=1)
        emotion_logits = self.emotion_classifier(pooled)
        emotion_probs = torch.sigmoid(emotion_logits)
        
        # Initialize decoder hidden state
        # Encoder hidden is [num_layers*2, batch, hidden_size] (bidirectional)
        # We need [num_layers, batch, hidden_size*2] for decoder
        # Reshape: combine forward and backward for each layer
        decoder_hidden = self._bridge_bidirectional_hidden(encoder_hidden)
        
        if tgt is not None:
            # Training mode
            max_len = tgt.size(1)
            decoder_input = tgt[:, 0]
            
            outputs = []
            attentions = []
            
            for t in range(1, max_len):
                output, decoder_hidden, attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, emotion_probs
                )
                outputs.append(output)
                attentions.append(attention)
                
                # Teacher forcing
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                decoder_input = tgt[:, t] if use_teacher_forcing else output.argmax(1)
            
            outputs = torch.stack(outputs, dim=1)
            attentions = torch.stack(attentions, dim=1)
        else:
            # Inference mode
            outputs = None
            attentions = None
        
        return outputs, emotion_logits, attentions
    
    def _bridge_bidirectional_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        Bridge bidirectional encoder hidden to decoder hidden.
        
        Args:
            encoder_hidden: [num_layers*2, batch, hidden_size]
        
        Returns:
            decoder_hidden: [num_layers, batch, hidden_size*2]
        """
        # encoder_hidden shape: [num_layers*2, batch, hidden_size]
        # Split into forward and backward
        batch_size = encoder_hidden.size(1)
        hidden_size = encoder_hidden.size(2)
        
        # Reshape to [num_layers, 2, batch, hidden_size]
        encoder_hidden = encoder_hidden.view(self.num_layers, 2, batch_size, hidden_size)
        
        # Concatenate forward and backward: [num_layers, batch, hidden_size*2]
        decoder_hidden = torch.cat([encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]], dim=2)
        
        return decoder_hidden
    
    def generate(self, src: torch.Tensor, emotion_vector: torch.Tensor,
                 max_length: int = 50, sos_token: int = 1, eos_token: int = 2) -> List[int]:
        """
        Generate response given source and emotion.
        
        Args:
            src: Source sequence [1, seq_len]
            emotion_vector: Emotion probabilities [1, num_emotions]
            max_length: Maximum generation length
            sos_token: Start of sequence token
            eos_token: End of sequence token
        
        Returns:
            generated: List of generated token indices
        """
        self.eval()
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(src)
            decoder_hidden = self._bridge_bidirectional_hidden(encoder_hidden)
            
            decoder_input = torch.LongTensor([[sos_token]]).to(src.device)
            generated = []
            
            for _ in range(max_length):
                output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, emotion_vector
                )
                
                predicted = output.argmax(1)
                token = predicted.item()
                
                if token == eos_token:
                    break
                
                generated.append(token)
                decoder_input = predicted.unsqueeze(0)
        
        return generated


class ReinforcementLearningTrainer:
    """Reinforcement learning trainer with policy gradients."""
    
    def __init__(self, model: EmotionChatbotModel, device: torch.device):
        self.model = model
        self.device = device
    
    def compute_reward(self, generated: List[int], target: torch.Tensor,
                      emotion_match: float) -> float:
        """
        Compute reward for generated sequence.
        
        Args:
            generated: Generated token sequence
            target: Target token sequence
            emotion_match: Emotion classification accuracy
        
        Returns:
            reward: Computed reward value
        """
        # Token-level accuracy
        target_list = target.cpu().tolist()
        min_len = min(len(generated), len(target_list))
        
        if min_len == 0:
            token_accuracy = 0.0
        else:
            matches = sum(1 for i in range(min_len) if generated[i] == target_list[i])
            token_accuracy = matches / min_len
        
        # Length penalty
        length_diff = abs(len(generated) - len(target_list))
        length_penalty = max(0, 1 - length_diff / max(len(target_list), 1))
        
        # Combined reward
        reward = 0.4 * token_accuracy + 0.3 * emotion_match + 0.3 * length_penalty
        
        return reward
    
    def policy_gradient_loss(self, log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute policy gradient loss.
        
        Args:
            log_probs: Log probabilities of actions
            rewards: Computed rewards
        
        Returns:
            loss: Policy gradient loss
        """
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Policy gradient loss
        loss = -(log_probs * rewards).mean()
        
        return loss


class EmotionChatbotTrainer:
    """Main trainer class for emotion chatbot."""
    
    def __init__(
        self,
        model: EmotionChatbotModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        model_dir: str = "models",
        use_rl: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.use_rl = use_rl
        
        # Optimizers
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.emotion_loss = nn.BCEWithLogitsLoss()
        
        # Reinforcement learning
        if use_rl:
            self.rl_trainer = ReinforcementLearningTrainer(model, device)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'emotion_accuracy': [],
            'generation_quality': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_emotion_loss = 0
        total_generation_loss = 0
        total_rl_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            text = batch['text'].to(self.device)
            emotion_labels = batch['emotion_labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            teacher_forcing_ratio = max(0.5, 1.0 - epoch / 20.0)  # Decay teacher forcing
            outputs, emotion_logits, _ = self.model(
                text, text, teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            # Compute losses
            # 1. Emotion classification loss
            emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
            
            # 2. Generation loss
            if outputs is not None:
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = text[:, 1:].reshape(-1)
                generation_loss = self.ce_loss(outputs_flat, targets_flat)
            else:
                generation_loss = torch.tensor(0.0).to(self.device)
            
            # 3. Reinforcement learning loss (every N batches)
            rl_loss = torch.tensor(0.0).to(self.device)
            if self.use_rl and batch_idx % 5 == 0:  # Apply RL every 5 batches
                rl_loss = self._compute_rl_loss(text, emotion_labels)
            
            # Combined loss
            loss = emotion_loss + generation_loss + 0.1 * rl_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_emotion_loss += emotion_loss.item()
            total_generation_loss += generation_loss.item()
            total_rl_loss += rl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'emotion': f'{emotion_loss.item():.4f}',
                'gen': f'{generation_loss.item():.4f}'
            })
        
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'emotion_loss': total_emotion_loss / len(self.train_loader),
            'generation_loss': total_generation_loss / len(self.train_loader),
            'rl_loss': total_rl_loss / len(self.train_loader)
        }
        
        return metrics
    
    def _compute_rl_loss(self, text: torch.Tensor, emotion_labels: torch.Tensor) -> torch.Tensor:
        """Compute reinforcement learning loss."""
        batch_size = text.size(0)
        rl_loss = 0
        
        for i in range(min(4, batch_size)):  # Sample subset for efficiency
            src = text[i:i+1]
            tgt = text[i:i+1]
            emotion = emotion_labels[i:i+1]
            
            # Generate response
            generated = self.model.generate(
                src, emotion, max_length=text.size(1),
                sos_token=1, eos_token=2
            )
            
            # Compute reward
            _, emotion_logits, _ = self.model(src, tgt, teacher_forcing_ratio=0)
            emotion_pred = torch.sigmoid(emotion_logits)
            emotion_match = F.cosine_similarity(emotion_pred, emotion).item()
            
            reward = self.rl_trainer.compute_reward(generated, tgt[0], emotion_match)
            
            # Accumulate RL loss (simplified policy gradient)
            rl_loss += -reward
        
        return torch.tensor(rl_loss / min(4, batch_size)).to(self.device)
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_emotion_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                text = batch['text'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                outputs, emotion_logits, _ = self.model(text, text, teacher_forcing_ratio=0)
                
                # Emotion loss
                emotion_loss = self.emotion_loss(emotion_logits, emotion_labels)
                
                # Generation loss
                if outputs is not None:
                    outputs_flat = outputs.reshape(-1, outputs.size(-1))
                    targets_flat = text[:, 1:].reshape(-1)
                    generation_loss = self.ce_loss(outputs_flat, targets_flat)
                else:
                    generation_loss = torch.tensor(0.0).to(self.device)
                
                loss = emotion_loss + generation_loss
                total_loss += loss.item()
                
                # Emotion accuracy
                emotion_pred = (torch.sigmoid(emotion_logits) > 0.5).float()
                correct = (emotion_pred == emotion_labels).all(dim=1).sum().item()
                total_emotion_correct += correct
                total_samples += text.size(0)
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'emotion_accuracy': total_emotion_correct / total_samples
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.model_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
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
            logger.info(f"  RL Loss: {train_metrics['rl_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Emotion Accuracy: {val_metrics['emotion_accuracy']:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Save metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['emotion_accuracy'].append(val_metrics['emotion_accuracy'])
            
            # Check for improvement
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
                # Save intermediate training plots
                self.plot_training_history()
            
            # Early stopping
            if self.patience_counter >= self.max_patience:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        logger.info("\nTraining completed!")
        
        # Plot and save training history
        self.plot_training_history()
        
        # Test and save results
        self.test()
    
    def test(self):
        """Test the model with comprehensive metrics and visualizations."""
        logger.info("\n" + "="*50)
        logger.info("Testing on test set...")
        logger.info("="*50)
        
        # Load best model
        best_model_path = self.model_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                text = batch['text'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                _, emotion_logits, _ = self.model(text, text, teacher_forcing_ratio=0)
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
        
        # Save results
        self._save_test_results(test_metrics, all_predictions, all_labels)
        
        # Plot confusion matrices
        self._plot_confusion_matrices(all_predictions, all_labels)
        
        logger.info(f"\nTest Results Saved to: {self.results_dir}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1 Score (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"Test Precision (Macro): {test_metrics['precision_macro']:.4f}")
        logger.info(f"Test Recall (Macro): {test_metrics['recall_macro']:.4f}")
        
        return test_metrics
    
    def _compute_test_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                              probs: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive test metrics."""
        # Overall accuracy (exact match)
        exact_match = (predictions == labels).all(axis=1).mean()
        
        # Hamming accuracy (label-wise)
        hamming_acc = (predictions == labels).mean()
        
        # For multi-label classification
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
        
        recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
        per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)
        
        metrics = {
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
        
        return metrics
    
    def _save_test_results(self, metrics: Dict, predictions: np.ndarray, labels: np.ndarray):
        """Save test results to file."""
        # Save metrics as JSON
        import json
        metrics_file = self.results_dir / 'test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved test metrics to {metrics_file}")
        
        # Save detailed classification report
        report_file = self.results_dir / 'classification_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EMOTION CHATBOT - TEST SET EVALUATION\n")
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
            
            # Get emotion names from train dataset
            emotion_names = sorted(self.train_loader.dataset.emotion_to_idx.keys())
            for i, emotion in enumerate(emotion_names):
                f1 = metrics['per_class_f1'][i]
                prec = metrics['per_class_precision'][i]
                rec = metrics['per_class_recall'][i]
                f.write(f"{emotion:<20} {f1:>10.4f} {prec:>12.4f} {rec:>10.4f}\n")
        
        logger.info(f"Saved classification report to {report_file}")
    
    def _plot_confusion_matrices(self, predictions: np.ndarray, labels: np.ndarray):
        """Plot and save confusion matrices for top emotions."""
        emotion_to_idx = self.train_loader.dataset.emotion_to_idx
        emotion_names = sorted(emotion_to_idx.keys())
        
        # Get top 10 most frequent emotions
        emotion_counts = labels.sum(axis=0)
        top_emotion_indices = np.argsort(emotion_counts)[-10:][::-1]
        
        # Create confusion matrix for each top emotion
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
        confusion_file = self.results_dir / 'confusion_matrices_top10.png'
        plt.savefig(confusion_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrices to {confusion_file}")
    
    def plot_training_history(self):
        """Plot and save training history graphs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training and Validation Loss
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Emotion Accuracy
        axes[0, 1].plot(self.history['emotion_accuracy'], label='Validation Accuracy', 
                       color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Emotion Classification Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Loss Comparison Bar Chart (Last Epoch)
        if len(self.history['train_loss']) > 0:
            categories = ['Train Loss', 'Val Loss']
            values = [self.history['train_loss'][-1], self.history['val_loss'][-1]]
            axes[1, 0].bar(categories, values, color=['blue', 'orange'], alpha=0.7)
            axes[1, 0].set_ylabel('Loss', fontsize=12)
            axes[1, 0].set_title('Final Epoch Loss Comparison', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(values):
                axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
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
        history_file = self.results_dir / 'training_history.png'
        plt.savefig(history_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training history plots to {history_file}")
        
        # Save history as JSON
        history_json = self.results_dir / 'training_history.json'
        with open(history_json, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Saved training history data to {history_json}")


def main():
    """Main training function."""
    # Configuration
    config = {
        'train_data': r"C:\Users\sslue\AI chatbot\EmpathAI-Emotion-Chatbot\preprocessed_data_roberta\json\train.json",
        'val_data': r"C:\Users\sslue\AI chatbot\EmpathAI-Emotion-Chatbot\preprocessed_data_roberta\json\validation.json",
        'test_data': r"C:\Users\sslue\AI chatbot\EmpathAI-Emotion-Chatbot\preprocessed_data_roberta\json\test.json",
        'batch_size': 32,
        'embedding_dim': 256,
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'max_length': 128,
        'model_dir': 'emotion_chatbot_models',
        'use_rl': True
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA not available. Training will be VERY slow on CPU.")
        logger.warning("Install CUDA-enabled PyTorch for GPU acceleration.")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = EmotionChatbotDataset(config['train_data'], max_length=config['max_length'])
    val_dataset = EmotionChatbotDataset(config['val_data'], vocab=train_dataset.vocab, max_length=config['max_length'])
    test_dataset = EmotionChatbotDataset(config['test_data'], vocab=train_dataset.vocab, max_length=config['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Create model
    logger.info("Creating model...")
    model = EmotionChatbotModel(
        vocab_size=len(train_dataset.vocab),
        num_emotions=len(train_dataset.emotion_to_idx),
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = EmotionChatbotTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        model_dir=config['model_dir'],
        use_rl=config['use_rl']
    )
    
    # Train
    trainer.train()
    
    logger.info("\n" + "="*50)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()
