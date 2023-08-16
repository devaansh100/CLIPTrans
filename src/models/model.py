import sys
sys.path.append('src/utils')
import torch
import torch.nn as nn
import transformers
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from clip_comb import *
from adapters import *
from model_utils import *
import warnings

from transformers.models.mbart.modeling_mbart import shift_tokens_right

'''
	Changes made in huggingface to: 
		src/transformers/generation/utils.py [_update_model_kwargs_for_generation, _expand_inputs_for_generation]
		src/transformers/models/mbart/modeling_mbart.py [prepare_decoder_attention_mask, prepare_inputs_for_generation]
'''
class CLIPTrans(nn.Module):
	def __init__(self, params):
		super(CLIPTrans, self).__init__()
		if params.image_encoder == 'mclip':
			print('Using MCLIP encoder')
			self.clip = M_CLIP(params)
		elif params.image_encoder == 'clip':
			print('Using CLIP encoder')
			self.clip = CLIP(params)
		elif params.image_encoder == 'clip_res':
			print('Using CLIP-Resnet encoder')
			self.clip = CLIP_RES(params)
		self.prefix_length = params.prefix_length
		# self.fusion = params.fusion
		if self.prefix_length > 0:
			adapter_inputs = {'clip_dim': 512, 'mbart_dim': 1024, 'prefix_length': self.prefix_length}
			if params.mapping_network == 'mlp':
				self.adapter = MLPAdapter(**adapter_inputs)
			elif params.mapping_network == 'transformer':
				self.adapter = TransformerAdapter(num_encoder_layers = 1, **adapter_inputs)
			else:
				self.adapter = HiddenMLPAdapter(**adapter_inputs)
		if self.prefix_length not in [0, 10]:# and self.fusion == 'context':
			warnings.warn("prefix_length != 10 or != 0. Change the combined attention_mask line in huggingface accordingly")
		self.tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
		# if self.fusion == 'gated_fusion':
		# 	self.mbart = MBartForConditionalGeneration_GF.from_pretrained('facebook/mbart-large-50')
		# 	setattr(self.mbart.model, 'get_gated_outputs', self.mbart.get_gated_outputs)
		# else:
		self.mbart = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
		self.target_lang = get_lang_code(params.tgt_lang)
		if not params.test:
			if params.stage in ['caption', 'text_recon']:
				freeze_params(self.mbart.model.encoder)
	
	def forward(self, batch, mode = 'train'): # TODO: Refactor to include dummy tokens and adapter
		if self.prefix_length > 0:
			visual_context = batch.pop('clip')
			prefix_embeds = self.adapter(visual_context)
		mask = batch['mbart'].pop('mask_decoder_input_ids') # Need to forcefully remove this since input embeds is being passed
		if mode == 'train':
			if self.prefix_length > 0:
				# From the labels, create decoder_input_ids
				# Ref: https://github.com/huggingface/transformers/blob/9e40bba6ba177cbc3b72b5fc7c8939174ad77899/src/transformers/models/mbart/modeling_mbart.py#L62
				# Masking decoder_input_ids so the same words as input_ids are present
				decoder_input_ids = shift_tokens_right(batch['mbart']['labels'], self.tokenizer.pad_token_id)
				# From the decoder_input_ids, get decoder_inputs_embeds
				# Ref: https://github.com/huggingface/transformers/blob/9e40bba6ba177cbc3b72b5fc7c8939174ad77899/src/transformers/models/mbart/modeling_mbart.py#L1027
				decoder_inputs_embeds = self.mbart.model.decoder.embed_tokens(decoder_input_ids) * self.mbart.model.decoder.embed_scale
				# Concat prefix_embeds and decoder_inputs_embeds in the format </s> O O O <LID> A B C </s>
				decoder_inputs_embeds = torch.cat((decoder_inputs_embeds[:, 0].unsqueeze(1), prefix_embeds, decoder_inputs_embeds[:, 1:]), dim = 1)
				batch['mbart']['decoder_inputs_embeds'] = decoder_inputs_embeds
				# Set decoder_attenion_mask accordingly
				decoder_attention_mask = torch.ones_like(decoder_input_ids).cuda()
				decoder_attention_mask[decoder_input_ids == 1] = 0
				decoder_attention_mask = torch.cat((decoder_attention_mask[:, 0].unsqueeze(1), torch.ones(prefix_embeds.shape[0], self.prefix_length).cuda(), decoder_attention_mask[:, 1:]), dim = 1)
				batch['mbart']['decoder_attention_mask'] = decoder_attention_mask
				# Modify the labels to accomdate prefix_embeds
				dummy_tokens = (torch.ones((batch['mbart']['labels'].shape[0], self.prefix_length)) * 1).cuda() # Ones because padding token in mbart is 1. # TODO: Confirm that GPT2 padding token is 0
				batch['mbart']['labels'] = torch.cat((batch['mbart']['labels'][:, 0].unsqueeze(1), dummy_tokens, batch['mbart']['labels'][:, 1:]), dim = 1).long()
				# elif self.fusion == 'gated_fusion':
				# 	batch['mbart']['clip_embs'] = prefix_embeds
			batch['mbart']['labels'][batch['mbart']['labels'] == 1] = -100 # To ignore loss on padding tokens
			outputs = self.mbart(**batch['mbart'])
		elif mode == 'test':
			batch['mbart'].pop('labels')
			if self.prefix_length > 0:
				# if self.fusion == 'context':
				decoder_input_ids = (torch.ones(batch['mbart']['input_ids'].shape[0], 1) * 2).long().cuda()
				decoder_inputs_embeds = self.mbart.model.decoder.embed_tokens(decoder_input_ids) * self.mbart.model.decoder.embed_scale
				decoder_inputs_embeds = torch.cat((prefix_embeds, decoder_inputs_embeds), dim = 1) # Putting prefix_embeds before eos, because for decoder, eos is now start_token_id
				batch['mbart']['decoder_inputs_embeds'] = decoder_inputs_embeds
				batch['mbart']['decoder_attention_mask'] = torch.ones((decoder_inputs_embeds.shape[0], decoder_inputs_embeds.shape[1])).cuda()
				# else:
				# 	batch['mbart']['clip_embs'] = prefix_embeds
			outputs = self.mbart.generate(**batch['mbart'], forced_bos_token_id = self.tokenizer.lang_code_to_id[self.target_lang], max_new_tokens = 60)
		return outputs


class CLIPTrans_CLIP(CLIPTrans):
	def __init__(self, params):
		super().__init__(params)
		# Ideally you could delete the image encoder but sometimes the image and text encoder is not separated(check clip_comb.py)
		# Besides, keeping the weights in memory barely affects performance and removing it require separate functions in each.
	
	def forward(self, batch, mode = 'train'):
		batch['clip'] = self.clip(batch['clip'])
		return super().forward(batch, mode)
