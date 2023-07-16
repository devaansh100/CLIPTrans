import sys
sys.path.append('..')

import torch
from tqdm import tqdm
import pickle as pkl
import os
from data_utils import *
from collate_fns import *
from dataset import DocDataset
from torch.utils.data import DataLoader
import warnings
from ddp import *

def get_WIT(params, model, test_ds = ['val'], force_pretraining = False):
	test_ds = test_ds[0]
	langs = [params.src_lang, params.tgt_lang]
	datapath = os.path.join(params.data_dir, f'wit/mmt/{params.src_lang}_{params.tgt_lang}')
	os.makedirs(os.path.join(datapath, 'mbart'), exist_ok = True)
	os.makedirs(os.path.join(datapath, params.image_encoder), exist_ok = True)
	# Reading train files
	train_texts = {lang: open(os.path.join(datapath, f'train.{lang}')).read().splitlines() for lang in langs}
	try:
		train_tok_mbart = {lang: pkl.load(open(os.path.join(datapath, f'mbart/train.{lang}.pkl'), 'rb')) for lang in langs}
	except:
		print('Did not find mbart train tokenized data. Creating...')
		train_tok_mbart = {lang: tokenize(train_texts[lang], model.tokenizer, lang, os.path.join(datapath, f'mbart/train.{lang}.pkl'), f'Tokenizing train {lang} with mbart') for lang in langs}

	try:
		train_tok_mclip = {lang: pkl.load(open(os.path.join(datapath, f'{params.image_encoder}/train.{lang}.pkl'), 'rb')) for lang in langs}
	except:
		print('Did not find mclip train tokenized data. Creating...')
		train_tok_mclip = {lang: tokenize(train_texts[lang], model.clip.text_preprocessor, lang, os.path.join(datapath, f'{params.image_encoder}/train.{lang}.pkl'), f'Tokenizing train {lang} with {params.image_encoder}') for lang in langs}

	# Reading test files
	test_texts = {lang: open(os.path.join(datapath, f'{test_ds}.{lang}')).read().splitlines() for lang in langs}
	try:
		test_tok_mbart = {lang: pkl.load(open(os.path.join(datapath, f'mbart/{test_ds}.{lang}.pkl'), 'rb')) for lang in langs}
	except:
		print('Did not find mbart test tokenized data. Creating...')
		test_tok_mbart = {lang: tokenize(test_texts[lang], model.tokenizer, lang, os.path.join(datapath, f'mbart/{test_ds}.{lang}.pkl'), f'Tokenizing test {lang} with mbart') for lang in langs}

	try:
		test_tok_mclip = {lang: pkl.load(open(os.path.join(datapath, f'{params.image_encoder}/{test_ds}.{lang}.pkl'), 'rb')) for lang in langs}
	except:
		print('Did not find mclip test tokenized data. Creating...')
		test_tok_mclip = {lang: tokenize(test_texts[lang], model.clip.text_preprocessor, lang, os.path.join(datapath, f'{params.image_encoder}/{test_ds}.{lang}.pkl'), f'Tokenizing test {lang} with {params.image_encoder}') for lang in langs}

	train_image_splits = open(os.path.join(datapath, f'train_image.txt')).read().splitlines()
	test_image_splits = open(os.path.join(datapath, f'{test_ds}_image.txt')).read().splitlines()
	# Getting images and embedding with CLIP. Same for text
	print('Loaded all text files. Getting images...')
	train_ignore_indices = [int(x) for x in train_image_splits[0].split(',')]
	train_img_embs = get_image_embs(model.clip, os.path.join(datapath, 'images'), train_image_splits[1:], os.path.join(datapath, f'{params.image_encoder}/train.pth'), 'Embedding train images', model.clip.image_preprocessor, train_ignore_indices)
	train_ignore_indices_f = os.path.join(datapath, 'images/train_ignore_indices.pkl')
	if not os.path.exists(train_ignore_indices_f):
		with open(train_ignore_indices_f, 'wb') as f:
			pkl.dump(train_ignore_indices, f)
		warnings.warn('train_ignore_indices.pkl was created. If the images were not read and embedded prior to this, this is an error. You would need to preprocess the image data again, ' +  
					  'since train_ignore_indices.pkl is not correctly aligned since it is the same as what is in train_image.txt, but does not include indices which failed to get read by PIL.')
	else:
		with open(train_ignore_indices_f, 'rb') as f:
			train_ignore_indices = pkl.load(f)

	test_ignore_indices = [int(x) for x in test_image_splits[0].split(',')]
	test_img_embs = get_image_embs(model.clip, os.path.join(datapath, 'images'), test_image_splits[1:], os.path.join(datapath, f'{params.image_encoder}/{test_ds}.pth'), f'Embedding {test_ds} images', model.clip.image_preprocessor, test_ignore_indices)
	test_ignore_indices_f = os.path.join(datapath, f'images/{test_ds}_ignore_indices.pkl')
	if not os.path.exists(test_ignore_indices_f):
		with open(test_ignore_indices_f, 'wb') as f:
			pkl.dump(test_ignore_indices, f)
		warnings.warn(f'{test_ds}_ignore_indices.pkl was created. If the images were not read and embedded prior to this, this is an error. You would need to preprocess the image data again, ' +  
					  f'since {test_ds}_ignore_indices.pkl is not correctly aligned since it is the same as what is in {test_ds}_image.txt, but does not include indices which failed to get read by PIL.')
	else:
		with open(test_ignore_indices_f, 'rb') as f:
			test_ignore_indices = pkl.load(f)
	
	train_text_embs, test_text_embs = {}, {}
	for lang in langs:
		embs_f = os.path.join(datapath, f'{params.image_encoder}/train.{lang}.pth')
		try:
			train_text_embs[lang] = torch.load(embs_f)
		except:
			text_ds = DocDataset(train_tok_mclip[lang])
			text_dl = DataLoader(text_ds, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True, collate_fn = collate_texts)
			train_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding train.{lang} mclip')

		embs_f = os.path.join(datapath, f'{params.image_encoder}/{test_ds}.{lang}.pth')
		try:
			test_text_embs[lang] = torch.load(embs_f)
		except:
			text_ds = DocDataset(test_tok_mclip[lang])
			text_dl = DataLoader(text_ds, batch_size = 128, shuffle = False, num_workers = 4, pin_memory = True, collate_fn = collate_texts)
			test_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding {test_ds}.{lang} mclip')

	return postprocess_pairs(train_texts, test_texts, train_tok_mbart, test_tok_mbart, train_img_embs, test_img_embs, train_text_embs, test_text_embs, params, train_ignore_indices, test_ignore_indices, train_tok_mclip, test_tok_mclip, force_pretraining)