import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
import zipfile
from glob import glob
from tqdm import tqdm

import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e10



# UNZIP
# maestro_zip = zipfile.ZipFile('/home/dasol/userdata/onsets-and-frames/maestro-v3.0.0.zip')
# maestro_zip.extractall('/home/dasol/userdata/onsets-and-frames/maestro')
# maestro_zip.close()


maestro_path = Path('/home/dasol/userdata/onsets-and-frames/maestro/maestro-v3.0.0')


def time2frame(time, sr=16000, hop_length=512):
    return round((time * sr) / hop_length)


def prepare_roll(path):
    sr = 16000
    hop_length = 512
    frame_per_sec =  sr/hop_length

    # make label - total
    midi = pretty_midi.PrettyMIDI(midi_file=path)
    total = midi.get_piano_roll(fs=frame_per_sec, times=None, pedal_threshold=64)
    total[total > 0] = 1 # remove velocity
    total = total[21:109]

    # make label - onset
    onset = np.zeros_like(total)
    for note in midi.instruments[0].notes:
        onset[int(note.pitch) - 21, int(note.start * frame_per_sec) - 1] = 1 # IndexError: index 10231 is out of bounds for axis 1 with size 10231
    pianoroll = np.stack([total, onset], axis=0)
    
    return pianoroll


def prepare_maestro(dir_path):
    metadata = pd.read_json(dir_path / 'maestro-v3.0.0.json')
    split, midi_filename = list(metadata['split']), list(metadata['midi_filename'])
    data_info = []
    for idx in range(len(split)):
        data_info.append((split[idx], midi_filename[idx])) 

    train_path = Path('/home/dasol/userdata/onsets-and-frames/datasets/maestro/train')
    test_path = Path('/home/dasol/userdata/onsets-and-frames/datasets/maestro/test')
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    missing_list = []

    for info in tqdm(data_info):
        target_path = train_path if info[0] == 'train' else test_path
        midi_path = dir_path / info[1]
        wav_path = midi_path.with_suffix('.wav')
        if not wav_path.exists() :
            missing_list.append(wav_path)
            continue
        
        roll = prepare_roll(str(midi_path))
        audio, sr = torchaudio.load(wav_path)
        audio = torchaudio.functional.resample(audio, sr, 16000).mean(dim=0) # stereo to mono     
        filename = (target_path / midi_path.stem).with_suffix('.pt') 
        torch.save({'audio':audio, 'label':np.array(roll, dtype=bool)}, filename) # float to bool

    print(missing_list)


prepare_maestro(maestro_path)
