import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders.wenet.transformer.encoder_cat import ConformerEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
from torchaudio.transforms import MelSpectrogram

class AudioPreEmphasis(nn.Module):

    def __init__(self, coeff=0.97):
        super().__init__()

        self.w = torch.FloatTensor([-coeff, 1.0]).unsqueeze(0).unsqueeze(0)

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        print("For error detection:",audio.shape)
        audio = F.pad(audio, (1, 0), 'reflect')
        return F.conv1d(audio, self.w.to(audio.device)).squeeze(1)

class SAP(nn.Module):

    def __init__(self, out_size, dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(out_size, dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, out_size, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, X):
        b, c, h, w = X.size()

        X = X.reshape(b, -1, w)
        #print(X.shape)
        W = self.attention(X)
        return torch.sum(W * X, dim=2)

class Conformer(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=1024, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos"):

        super(Conformer, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        
        self.features_extractor = nn.Sequential(
            AudioPreEmphasis(),
            MelSpectrogram(
                n_fft=512,
                win_length=400,
                hop_length=160,
                window_fn=torch.hamming_window,
                n_mels=n_mels
            )
        )
        self.instance_norm = nn.InstanceNorm1d(n_mels)
        
        self.conformer = ConformerEncoder(input_size=n_mels, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
        self.pooling = AttentiveStatisticsPooling(output_size*num_blocks)
        self.bn = BatchNorm1d(input_size=output_size*num_blocks*2)
        self.fc = torch.nn.Linear(output_size*num_blocks*2, embedding_dim)
    
    def forward(self, feat):
        feat = self.features_extractor(feat) + 1e-6

        feat = feat.log()
        feat = self.instance_norm(feat)

        feat = feat.unsqueeze(1)
        
        feat = feat.squeeze(1).permute(0, 2, 1)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens*feat.shape[1]).int()
        x, masks = self.conformer(feat, lens)
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

def conformer_cat(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=192, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
    model = Conformer(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size, 
            embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model

 
