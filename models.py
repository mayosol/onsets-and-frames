import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm



class BaselineModel(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                                n_fft = 2048,
                                                                hop_length = 512,
                                                                f_min = 20,
                                                                f_max = 8000,
                                                                n_mels = 88 * 4,
                                                            )
        self.rnn = nn.GRU(88*4, hidden_size, num_layers=4, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(hidden_size * 2, 88)

    def forward(self, x):
        mel_spec = self.mel_converter(x) # batch size x num mels x time
        hidden_out, last_hidden = self.rnn(mel_spec.permute(0, 2, 1))
        logit = self.projection(hidden_out)
        prob = torch.sigmoid(logit)
        return prob.permute(0, 2, 1) # pianoroll 형식으로 변환



class CNNModel(BaselineModel):
    def __init__(self, hidden_size=256):
        super().__init__(hidden_size)
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
                                 
        self.fc = nn.Linear(88 * hidden_size, hidden_size) 
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=3, bidirectional=True, batch_first=True)

    def forward(self, x):
        mel_spec = self.mel_converter(x) # batch size x num mels x time
        mel_spec = mel_spec.unsqueeze(1)
        conv_out = self.cnn(mel_spec)
        conv_out = conv_out.reshape(x.shape[0], -1, mel_spec.shape[-1])
        fc_out = self.fc(conv_out.permute(0, 2, 1))
        hidden_out, last_hidden = self.rnn(fc_out)
        logit = self.projection(hidden_out)
        prob = torch.sigmoid(logit)
        return prob.permute(0, 2, 1) # pianoroll 형식으로 변환



def binary_cross_entropy_loss(pred, target, eps=1e-7):
    return -(target * torch.log(pred+eps) + (1-target) * torch.log((1-pred)+eps)).mean()

def compute_metric(pred):
    cross_entropy = 