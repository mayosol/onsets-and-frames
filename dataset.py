import numpy as np
from pathlib import Path
import random

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e10



def time2frame(time, hop_length=512):
    return round(time / (hop_length / 1000))

class MAPSDataset:
    def __init__(self, dir_path, split, sr=16000):
        self.sr = sr
        self.hop_length = 512
        self.dir_path = list(dir_path.glob('./*/*/'))
        self.frame_per_sec =  self.sr/self.hop_length

        # split train, test set
        train_folder = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']
        test_folder = ['ENSTDkAm', 'ENSTDkCl']
        self.target_folder = train_folder if split=='train' else test_folder

        self._load_data()

    def get_piano_roll(self, path):
        # make label - total
        midi = pretty_midi.PrettyMIDI(midi_file=path)
        total = midi.get_piano_roll(fs=self.frame_per_sec, times=None, pedal_threshold=64)
        total[total > 0] = 1 # remove velocity
        total = total[21:109]
        # make label - onset
        onset = np.zeros_like(total)
        for note in midi.instruments[0].notes:
            onset[int(note.pitch) - 21, int(note.start * self.frame_per_sec)] = 1
        pianoroll = np.stack([total, onset], axis=0)
        return pianoroll

    def _load_data(self):
        # make lists of wav path, pianoroll
        self.audio = []
        self.roll = []
        for path in self.dir_path:
            folder = path.parent.name
            wav_files = list(path.rglob('*.wav'))
            missing_list = []

            if folder in self.target_folder:
                for file in wav_files:
                    midi = file.with_suffix('.mid')
                    if not midi.exists():
                        missing_list.append(id)
                        continue
                    roll = self.get_piano_roll(str(midi))
                    self.audio.append(file)
                    self.roll.append(roll)
        print(missing_list)

    def __len__(self):
        return len(self.audio)
        
    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.audio[idx])
        roll = self.roll[idx]
        start = random.randint(0, len(audio[0]) - (self.sr * 25) - 1)
        end = start + (self.sr * 20)
        sliced_audio = audio[:, start:end].mean(dim=0) # stereo to mono

        start_roll = time2frame(start/self.sr)
        end_roll = start_roll + int(20 * self.sr/512)
        sliced_roll = roll[:, :, start_roll:end_roll]
        return sliced_audio, sliced_roll

