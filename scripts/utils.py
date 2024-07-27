import torch
import librosa
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor

class SpeechDataset(Dataset):
    def __init__(self, audio_paths, transcriptions, processor):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_input, _ = librosa.load(self.audio_paths[idx], sr=16000)
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": self.transcriptions[idx]
        }

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df['audio_path'].tolist(), df['transcription'].tolist()

def create_data_loader(dataset, batch_size, sampler):
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))
    return data_loader
