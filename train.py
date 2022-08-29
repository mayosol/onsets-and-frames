import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import argparse

import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from glob import glob

import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e10

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 32 // 1000
MIN_MIDI = 21
MAX_MIDI = 108

N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048

DEV = "cuda"


def time2frame(time, sr=16000, hop_length=512):
    return round((time * sr) / hop_length)


class Dataset:
    def __init__(self, path, split):
        train_path = path / 'train'
        test_path = path / 'test'
        self.path = train_path if split == 'train' else test_path
        self.sample_rate = 16000

        self.files = list(self.path.glob('*.pt'))
        self.data = [torch.load(x) for x in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.data[idx]
        audio = data['audio']
        roll = np.array(data['label'], dtype=np.float32)

        start = random.randint(0, len(audio[0]) - (self.sample_rate * 25) - 1)
        end = start + (self.sample_rate * 20)
        sliced_audio = audio[start:end].mean(dim=0) # stereo to mono

        start_roll = time2frame(start/self.sample_rate)
        end_roll = start_roll + int(20 * self.sample_rate/512)
        sliced_roll = roll[:, :, start_roll:end_roll]

        return sliced_audio, sliced_roll   


def pitch2hz(pitch):
    return 2 ** ((pitch-69) / 12) * 440


class ConvStack(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, hidden_size//4, kernel_size=(3, 3), padding=1),
                                 nn.BatchNorm2d(hidden_size//4),
                                 nn.MaxPool2d((2, 1)),
                                 nn.ReLU(),
                                 
                                 nn.Conv2d(hidden_size//4, hidden_size//2, kernel_size=(3, 3), padding=1),
                                 nn.BatchNorm2d(hidden_size//2),
                                 nn.MaxPool2d((2, 1)),
                                 nn.ReLU(),
                                 
                                 nn.Conv2d(hidden_size//2, hidden_size, kernel_size=(3, 3), padding=1),
                                 nn.BatchNorm2d(hidden_size),
                                 nn.ReLU(),)
                                 
        self.fc = nn.Sequential(
            nn.Linear(88 * hidden_size, hidden_size),
            nn.Dropout(0.5)
        )

    def forward(self, mel_spec):
        out = self.cnn(mel_spec)
        out = out.reshape(mel_spec.shape[0], -1, mel_spec.shape[-1])
        out = self.fc(out.permute(0, 2, 1))
        return out


class BiLSTM(nn.Module):
    def __init__(self, input_features=256, hidden_size=128):
        super().__init__()
        self.rnn = nn.LSTM(input_features, hidden_size, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


class OnsetandFrameModel(nn.Module):
    def __init__(self): 
        super().__init__()
        
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                                  n_fft = 2048,
                                                                  hop_length = 512,
                                                                  f_min = 20,
                                                                  f_max = 8000,
                                                                  n_mels = 88 * 4)

        # sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(),
            BiLSTM(),
            nn.Linear(256, 88),
            nn.Sigmoid()
        ).cuda()
        self.frame_stack = nn.Sequential(
            ConvStack(),
            nn.Linear(256, 88),
            nn.Sigmoid()
        ).cuda()
        self.combined_stack = nn.Sequential(
            BiLSTM(176, 128), # ?
            nn.Linear(256, 88),
            nn.Sigmoid()
        ).cuda()

    def forward(self, x):
        mel_spec = self.mel_converter(x).unsqueeze(1).to(device=DEV)
        onset_pred = self.onset_stack(mel_spec)
        # offset_pred = self.offset_stack(mel_spec)
        activation_pred = self.frame_stack(mel_spec)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        # velocity_pred = self.velocity_stack(mel_spec)
        return onset_pred.permute(0, 2, 1), activation_pred.permute(0, 2, 1), frame_pred.permute(0, 2, 1)
        

def f1_score(threshold, roll):
    true_positive = ((threshold==1) * (roll==1)).sum()
    false_positive = ((threshold==1) * (roll==0)).sum()
    false_negative = ((threshold==0) * (roll==1)).sum()

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1_score = 2 / ((1 / precision) + (1 / recall))

    return f1_score



def main():
    parser = argparse.ArgumentParser()
    parser.add_argment('dir_path')
    args = parser.parse_args()
    dir_path = args.dir_path
    
    trainset = Dataset(dir_path, 'train')
    testset = Dataset(dir_path, 'test')

    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=10, num_workers=2)

    model = OnsetandFrameModel().to(device=DEFAULT_DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    BCEloss = nn.BCELoss()
    num_epochs = 50
    loss_record = []
    test_record = []
    f1_record = []
    model.train()

    for epoch in  tqdm(range(num_epochs)):
        for batch in tqdm(train_loader, leave=False):
            audio, roll = batch
            onset_pred, activation_pred, frame_pred = model(audio.to(device=DEFAULT_DEVICE))
            onset_loss = BCEloss(onset_pred[..., :-1].to(torch.float32), roll[:, 1].to(DEFAULT_DEVICE).to(torch.float32))
            frame_loss = BCEloss(frame_pred[..., :-1].to(torch.float32), roll[:, 0].to(DEFAULT_DEVICE).to(torch.float32))
            loss = onset_loss + frame_loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            loss_record.append(loss.item())

        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            for batch in test_loader:
                audio, roll = batch
                onset_test_pred, activation_test_pred, frame_test_pred = model(audio.to(device=DEFAULT_DEVICE))
                onset_test_loss = BCEloss(onset_test_pred[..., :-1].to(torch.float32), roll[:, 1].to(DEFAULT_DEVICE).to(torch.float32))
                frame_test_loss = BCEloss(frame_test_pred[..., :-1].to(torch.float32), roll[:, 0].to(DEFAULT_DEVICE).to(torch.float32))
                test_loss = onset_test_loss + frame_test_loss
                epoch_loss += test_loss.item()

                # TODO : Calculate accuracy (F1 score)
                roll = roll.to(torch.int).to(DEFAULT_DEVICE)
                onset_threshold = (onset_test_pred>=0.01).to(torch.int)[..., :-1]
                frame_threshold = (frame_test_pred>=0.1).to(torch.int)[..., :-1]

                onset_f1_score = f1_score(onset_threshold.to(DEFAULT_DEVICE), roll[:, 1])
                frame_f1_score = f1_score(frame_threshold.to(DEFAULT_DEVICE), roll[:, 0])
                f1_record.append((onset_f1_score + frame_f1_score) / 2)

            test_record.append(epoch_loss / len(test_loader))
        model.train()

if __name__ == "__main__":
    main()
