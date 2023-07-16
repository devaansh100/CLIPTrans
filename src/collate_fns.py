import torch
from torch.nn.utils.rnn import pad_sequence
import transformers

def collate_texts(batch): # Used with DocDataset for creating embeddings
	collate_util = lambda key, b, pad_val : pad_sequence([x[key].squeeze() for x in b], batch_first = True, padding_value = pad_val)
	input_ids = collate_util('input_ids', batch, 1)
	att_mask  = collate_util('attention_mask', batch, 0)
	return {'input_ids': input_ids, 'attention_mask': att_mask}

def collate_images(batch): # Used with DocDataset for creating embeddings
	return torch.stack(batch)

def collate_multi(batch): # Used with MultiModalDataset - calls above two since the format is the same
	out, labels, masks = {'clip': [], 'mbart': [], 'raw': []}, [], []
	for b in batch:
		out['clip'].append(b[0])
		out['mbart'].append(b[1])
		masks.append(b[2])
		labels.append(b[3])
		out['raw'].append(b[4])
		if len(b) > 5:
			out['clip'].append(b[5])
			out['mbart'].append(b[6])
			masks.append(b[7])
			labels.append(b[8])
			out['raw'].append(b[9])
	# breakpoint()
	out['clip'] = collate_images(out['clip']) if not isinstance(out['clip'][0], transformers.tokenization_utils_base.BatchEncoding) else collate_texts(out['clip'])
	out['mbart'] = collate_texts(out['mbart'])
	out['mbart']['labels'] = collate_texts(labels)['input_ids']
	out['mbart']['mask_decoder_input_ids'] = pad_sequence([x.squeeze() for x in masks], batch_first = True, padding_value = True)
	return out