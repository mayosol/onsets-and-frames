import numpy as np
import torch
import torchaudio
from pathlib import Path
import zipfile
from glob import glob

import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e10



# UNZIP
# maps_piano_path = list(maps_path.rglob('*_2.zip'))
# for path in maps_piano_path:
#     zipfile.ZipFile(path).extractall(maps_path)


maps_path = Path('/home/dasol/userdata/onsets-and-frames/maps')



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
        onset[int(note.pitch) - 21, int(note.start * frame_per_sec)] = 1
    pianoroll = np.stack([total, onset], axis=0)
    
    return pianoroll


def prepare_maps(dir_path):
    dir_path = list(dir_path.glob('./*/'))

    train_folder = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']
    test_folder = ['ENSTDkAm', 'ENSTDkCl']

    train_path = Path('/home/dasol/userdata/onsets-and-frames/datasets/maps/train')
    test_path = Path('/home/dasol/userdata/onsets-and-frames/datasets/maps/test')
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    missing_list = []
    
    for dir in dir_path:
        target_path = train_path if dir.name in train_folder else test_path
        print(target_path, dir.name)
        wav_path = list(dir.rglob('*.wav'))
        
        for path in wav_path:
            midi = path.with_suffix('.mid')
            if not midi.exists():
                missing_list.append(id)
                continue

            roll = prepare_roll(str(midi))
            audio, sr = torchaudio.load(path)
            audio = torchaudio.functional.resample(audio, sr, 16000)        
            filename = (target_path / path.stem).with_suffix('.pt')
    
            torch.save({'audio':audio, 'label':roll}, filename)

    print(missing_list)


prepare_maps(maps_path)
