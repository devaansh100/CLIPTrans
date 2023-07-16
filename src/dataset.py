import torch
from torch.utils.data import Dataset
import math

class DocDataset(Dataset):
	def __init__(self, docs):
		super().__init__()
		self.docs = docs

	def __len__(self):
		return len(self.docs)

	def __getitem__(self, idx):
		return self.docs[idx]

class ConcatDataset(Dataset):
	def __init__(self, dataset1, dataset2):
		super().__init__()
		len1 = len(dataset1)
		len2 = len(dataset2)
		if len1 > len2: # self.dataset1 is the longer dataset
			self.dataset1 = dataset1
			self.dataset2 = dataset2
			self.length = len2 + math.ceil((len1 - len2)/2)
		elif len1 < len2:
			self.dataset1 = dataset2
			self.dataset2 = dataset1
			self.length = len1 + math.ceil((len2 - len1)/2)
		else:
			self.dataset1 = dataset2
			self.dataset2 = dataset1
			self.length = len1
		self.len1 = len(self.dataset1)
		self.len2 = len(self.dataset2)
		
	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		if idx >= self.len2:
			idx2 = idx - self.len2 + 1
			item1 = self.dataset1[idx]
			if idx == self.len1 - idx2: # To avoid returning the same item twice in a batch
				return (*item1,)
			else:
				item2 = self.dataset1[-idx2]
				batch = item1 + item2
				return (*batch,)
		else:
			batch = self.dataset1[idx] + self.dataset2[idx]
			return (*batch,)

class MultiModalDataset(Dataset):
	def __init__(self, params, clip_embs, tok_data, raw_data, clip_tok_data, mask_token = None, mask_inputs = False, is_pretraining = False):
		super().__init__()
		self.is_pretraining = is_pretraining
		self.mask_prob = params.mask_prob
		self.mask_inputs = mask_inputs
		self.mask_token = mask_token
		self.use_clip_tok = params.unfreeze_clip
		if self.is_pretraining:
			self.input_tok  = tok_data
			self.output_raw = raw_data
			self.langs = list(self.input_tok.keys())
			self.clip_embs  = clip_embs
		else:
			self.input_tok = tok_data[params.src_lang]
			self.output_tok = tok_data[params.tgt_lang]
			self.output_raw = raw_data[params.tgt_lang]
			self.clip_embs  = clip_embs[params.src_lang] if isinstance(clip_embs, dict) else clip_embs
			self.clip_tok = clip_tok_data[params.src_lang]
		self.length = len(self.clip_embs[params.src_lang]) if isinstance(self.clip_embs, dict) else len(self.clip_embs)
		self.use_all_langs = params.use_all_langs
		if self.use_all_langs:
			self.single_length = self.length
			self.length *= len(tok_data)
			self.langs = list(tok_data.keys())
		else:
			langs = [params.src_lang, params.tgt_lang]
			self.caption_lang = 'en' if 'en' in langs else 'es'

	def __len__(self):
		return self.length
	
	def mask_inputs_(self, inputs):
		if self.mask_inputs:
			mask_inputs = torch.rand(inputs['input_ids'].shape) > self.mask_prob
			mask_inputs[0][0] = mask_inputs[0][-1] = True # Keep LID and EOS
			return {'input_ids': inputs['input_ids'][mask_inputs], 'attention_mask': inputs['attention_mask'][mask_inputs]}, mask_inputs
		return inputs, None
	
	def __getitem__(self, idx):
		batch = []
		if self.is_pretraining:
			if hasattr(self, 'caption_lang'):
				lang = self.caption_lang
			else:
				lang_idx = -1
				while idx >= self.single_length:
					idx -= self.single_length
					lang_idx -= 1
				lang = self.langs[lang_idx]

			if isinstance(self.clip_embs, dict):
				batch.append(self.clip_embs[lang][idx])
			else:
				batch.append(self.clip_embs[idx])
			inputs, mask = self.mask_inputs_(self.input_tok[lang][idx])
			batch.append(inputs)
			batch.append(mask)
			batch.append(self.input_tok[lang][idx])
			batch.append(self.output_raw[lang][idx])
		else:
			if self.use_clip_tok:
				batch.append(self.clip_tok[idx])
			else:
				batch.append(self.clip_embs[idx])
			inputs, _ = self.mask_inputs_(self.input_tok[idx])
			batch.append(inputs)
			outputs = self.output_tok[idx]
			mask = torch.ones_like(outputs['input_ids']).bool() # Mask doesnt do anything while finetuning. Only here to keep the collate_fn and forward same.
			batch.append(mask)
			batch.append(outputs)
			batch.append(self.output_raw[idx])
		return (*batch,)
