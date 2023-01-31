import torch
import torch.nn as nn
import torch.nn.functional as F

from sslsv.encoders import resnet, mae, utils


class ViT(nn.Module):
	def __init__(self, dataset='fsd50k', size='base', patch_size=None, c=True,
			use_learned_pos_embd=False, use_mean_pool=False, use_decoder=False):
		super().__init__()
		
		if patch_size is None:
			patch_size = [16, 16]
		if dataset == 'cifar10':
			self.encoder = mae.get_mae_vit(size, patch_size, c, use_learned_pos_embd=use_learned_pos_embd,
										img_size=(32,32), in_chans=3)
		else:
			self.encoder = mae.get_mae_vit(size, patch_size, c, use_learned_pos_embd=use_learned_pos_embd,
										use_decoder=use_decoder)
		
		self.embed_dim = self.encoder.embed_dim
		self.use_mean_pool = use_mean_pool

	def forward(self, x, mask_ratio=0, masked_recon=False):
		x = self.encoder(x, mask_ratio=mask_ratio, masked_recon=masked_recon,
					mean_pool=self.use_mean_pool)
		return x
