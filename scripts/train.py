import os
import torch
import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AdamW, get_scheduler
from datasets import load_metric
from torch.utils.data import RandomSampler
from utils import get_device, load_data, create_data_loader, SpeechDataset

def main(data_path):
    # Parameters
    model_name = 'facebook/wav2vec2-base-960h'
    batch_size = 4
    epochs = 3
    learning_rate = 3e-5

    # Load Dataset
    audio_paths, transcriptions = load_data(data_path)

    # Processor and Tokenizer
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # Tokenize Data
    transcriptions = processor.tokenizer(transcriptions, padding=True, truncation=True, return_tensors="pt").input_ids

    # Create Dataset and DataLoader
    dataset = SpeechDataset(audio_paths, transcriptions, processor)
    train_loader = create_data_loader(dataset, batch_size, RandomSampler)

    # Model
    device = get_device()
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training Function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data')
    args = parser.parse_args()
    main(args.data_path)
