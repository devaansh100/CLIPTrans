from multilingual_clip.pt_multilingual_clip import MultilingualCLIP
import transformers
import torch

class MClip(MultilingualCLIP):
	def __init__(self, config, *args, **kwargs):
		super().__init__(config, *args, **kwargs)

	def forward(self, txt_tok): # Removing on the fly tokenization
		embs = self.transformer(**txt_tok)[0]
		att = txt_tok['attention_mask']
		embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
		return self.LinearTransformation(embs)