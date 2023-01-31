import torch
import torch.nn as nn
import torch.nn.functional as F

from sslsv.encoders import resnet, mae, utils

from torchaudio.transforms import MelSpectrogram

class AudioPreEmphasis(nn.Module):

    def __init__(self, coeff=0.97):
        super().__init__()

        self.w = torch.FloatTensor([-coeff, 1.0]).unsqueeze(0).unsqueeze(0)

    def forward(self, audio):
        audio = audio.unsqueeze(1)
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
        #print(X.shape)

        X = X.reshape(b, -1, w)
        #print("1111",X.shape)
        W = self.attention(X)
        return torch.sum(W * X, dim=2)



class AudioNTT2022Encoder(nn.Module):
	"""
	Encoder network from BYOLA-v2
	Copy-paste from https://github.com/nttcslab/byol-a/blob/master/v2/byol_a2/models.py
	"""
	def __init__(self, n_mels=64, d=3072, base_d=64, mlp_hidden_d=2048, conv_layers=2, stack=True, squeeze_excitation=False):
		super().__init__()
		convs = [
			nn.Conv2d(1, base_d, 3, stride=1, padding=1),
			nn.BatchNorm2d(base_d),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
		]
		if squeeze_excitation:
			convs.append(SE_Block(c=base_d))
		for c in range(1, conv_layers):
			convs.extend([
				nn.Conv2d(base_d, base_d, 3, stride=1, padding=1),
				nn.BatchNorm2d(base_d),
				nn.ReLU(),
				nn.MaxPool2d(2, stride=2),
			])
			if squeeze_excitation:
				convs.append(SE_Block(c=base_d))
		self.features = nn.Sequential(*convs)
		self.conv_d = base_d * (n_mels//(2**conv_layers))
		self.fc = nn.Sequential(
			nn.Linear(self.conv_d, mlp_hidden_d),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(mlp_hidden_d, d - self.conv_d),
			nn.ReLU(),
		)
		self.stack = stack
		self.features_extractor = nn.Sequential(
			AudioPreEmphasis(),
			MelSpectrogram(
			n_fft=512,
			win_length=400,
			hop_length=160,
			window_fn=torch.hamming_window,
			n_mels = n_mels
		))
		self.instance_norm = nn.InstanceNorm1d(n_mels)

	def forward(self, x):
	
		x = self.features_extractor(x) + 1e-6
		x = x.log()
		x = self.instance_norm(x)
		x = x.unsqueeze(1)
		
		x = self.features(x)       # (batch, ch, mel, time)
		x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
		B, T, D, C = x.shape
		x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
		x_fc = self.fc(x)
		x = torch.hstack([x.transpose(1,2), x_fc.transpose(1,2)]).transpose(1,2) if self.stack else x_fc
		return x
class AudioNTT2022(AudioNTT2022Encoder):
	def __init__(self, n_mels=64, d=3072, mlp_hidden_d=2048, squeeze_excitation=False):
		super().__init__(n_mels=n_mels, d=d, mlp_hidden_d=mlp_hidden_d, squeeze_excitation=squeeze_excitation)
		self.embed_dim = d
		sap_out_size = int(n_mels / 8 * 256)
		self.sap = SAP(sap_out_size)
		self.fc = nn.Linear(sap_out_size, d)

	def forward(self, x):
		x = super().forward(x)
		#x = self.sap(x)
		#x = self.fc(x)
		#x = mean_max_pooling(x)
		return x


def mean_max_pooling(frame_embeddings):
	assert len(frame_embeddings.shape) == 3 # Batch,Time,Dimension
	(x1, _) = torch.max(frame_embeddings, dim=1)
	x2 = torch.mean(frame_embeddings, dim=1)
	x = x1 + x2
	return x

class SE_Block(nn.Module):
    """Copy-paste from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4 """
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
