import torch
from transformers import BertTokenizer, BertForSequenceClassification

class EmotionDetector:
    def __init__(self, model_name='google/goemotions', num_labels=27):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def predict(self, texts):
        inputs = self.preprocess(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.sigmoid(logits).cpu().numpy()
        return predictions

    def fine_tune(self, train_dataset, epochs=3, batch_size=16, learning_rate=5e-5):
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()