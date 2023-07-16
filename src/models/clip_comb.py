import torch
import torch.nn as nn
import transformers
import clip
from mclip import MClip
from model_utils import *
from transformers import AutoProcessor, AutoTokenizer, CLIPModel, BlipModel

class M_CLIP(nn.Module):
	def __init__(self, params):
		super(M_CLIP, self).__init__()
		self.text_encoder, self.text_preprocessor = MClip.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-32'), AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-32')
		self.image_encoder, self.image_preprocessor = CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
		if not params.unfreeze_clip:
			freeze_params(self.text_encoder)
		freeze_params(self.image_encoder)

	def forward(self, x):
		if isinstance(x, dict):
			return self.text_encoder(x)
		else:
			return self.image_encoder.get_image_features(x)

class CLIP(nn.Module):
	def __init__(self, params):
		super(CLIP, self).__init__()
		self.text_encoder, self.text_preprocessor = CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
		self.image_encoder, self.image_preprocessor = self.text_encoder, AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
		if not params.unfreeze_clip:
			freeze_params(self.text_encoder)
		freeze_params(self.image_encoder)

	def forward(self, x):
		if isinstance(x, dict):
			return self.text_encoder.get_text_features(**x)
		else:
			return self.image_encoder.get_image_features(**x)

class CLIP_RES(nn.Module):
	def __init__(self, params):
		super(CLIP_RES, self).__init__()
		self.image_encoder, self.image_preprocessor = clip.load('RN50x64')
		self.text_encoder, self.text_preprocessor = self.image_encoder, AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
		if not params.unfreeze_clip:
			freeze_params(self.text_encoder)
		freeze_params(self.image_encoder)

	def forward(self, x):
		if isinstance(x, dict):
			return self.text_encoder.encode_image(x)
		else:
			return self.image_encoder.encode_text(x)
