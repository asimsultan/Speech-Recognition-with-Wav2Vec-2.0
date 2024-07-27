import torch
import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_metric
from torch.utils.data import SequentialSampler
from utils import get_device, load_data, create_data_loader, SpeechDataset
import os

def main(data_path):
    # Parameters
    model_dir = './models'
    batch_size = 4

    # Load Model and Processor
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    audio_paths, transcriptions = load_data(data_path)
    transcriptions = processor.tokenizer(transcriptions, padding=True, truncation=True, return_tensors="pt").input_ids

    # Create Dataset and DataLoader
    dataset = SpeechDataset(audio_paths, transcriptions, processor)
    test_loader = create_data_loader(dataset, batch_size, SequentialSampler)

    # Evaluation Function
    def evaluate(model, data_loader, device, processor):
        model.eval()
        total_preds = []
        total_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_values = batch["input_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_values=input_values, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                total_preds.extend(processor.batch_decode(preds))
                total_labels.extend(processor.batch_decode(labels))

        wer = load_metric("wer").compute(predictions=total_preds, references=total_labels)

        return wer

    # Evaluate
    wer = evaluate(model, test_loader, device, processor)
    print(f'Word Error Rate (WER): {wer}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data')
    args = parser.parse_args()
    main(args.data_path)