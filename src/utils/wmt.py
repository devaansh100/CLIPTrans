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
from ddp import *

def get_WMT(params, model, test_ds = ['val'], force_pretraining = False):
	test_ds = test_ds[0]
	langs = [params.src_lang, params.tgt_lang]
	datapath = os.path.join(params.data_dir, f'wmt/wmt_{params.src_lang}_{params.tgt_lang}')
	os.makedirs(os.path.join(datapath, f'mbart'), exist_ok = True)
	os.makedirs(os.path.join(datapath, f'{params.image_encoder}'), exist_ok = True)
	# Reading train files
	train_texts = {lang: open(os.path.join(datapath, f'train.{lang}')).read().splitlines() for lang in tqdm(langs, desc = 'Loading raw text files', disable = not is_main_process())}
	try:
		train_tok_mbart = {lang: pkl.load(open(os.path.join(datapath, f'mbart/train.{lang}.pkl'), 'rb')) for lang in tqdm(langs, desc = 'Loading mbart tokenized train files', disable = not is_main_process())}
	except:
		print('Did not find mbart train tokenized data. Creating...')
		train_tok_mbart = {lang: tokenize(train_texts[lang], model.tokenizer, lang, os.path.join(datapath, f'mbart/train.{lang}.pkl'), f'Tokenizing train {lang} with mbart') for lang in langs}

	try:
		train_tok_mclip = {lang: pkl.load(open(os.path.join(datapath, f'{params.image_encoder}/train.{lang}.pkl'), 'rb')) for lang in tqdm(langs, desc = 'Loading mclip tokenized train files', disable = not is_main_process())}
	except:
		print('Did not find mclip train tokenized data. Creating...')
		train_tok_mclip = {lang: tokenize(train_texts[lang], model.clip.text_preprocessor, lang, os.path.join(datapath, f'{params.image_encoder}/train.{lang}.pkl'), f'Tokenizing train {lang} with mclip') for lang in langs}

	# Reading test files
	test_texts = {lang: open(os.path.join(datapath, f'{test_ds}.{lang}')).read().splitlines() for lang in tqdm(langs, desc = 'Loading test raw files', disable = not is_main_process())}
	try:
		test_tok_mbart = {lang: pkl.load(open(os.path.join(datapath, f'mbart/{test_ds}.{lang}.pkl'), 'rb')) for lang in tqdm(langs, desc = 'Loading mbart tokenized test files', disable = not is_main_process())}
	except:
		print('Did not find mbart test tokenized data. Creating...')
		test_tok_mbart = {lang: tokenize(test_texts[lang], model.tokenizer, lang, os.path.join(datapath, f'mbart/{test_ds}.{lang}.pkl'), f'Tokenizing test {lang} with mbart') for lang in langs}

	try:
		test_tok_mclip = {lang: pkl.load(open(os.path.join(datapath, f'{params.image_encoder}/{test_ds}.{lang}.pkl'), 'rb')) for lang in tqdm(langs, desc = 'Loading mclip tokenized test files', disable = not is_main_process())}
	except:
		print('Did not find mclip test tokenized data. Creating...')
		test_tok_mclip = {lang: tokenize(test_texts[lang], model.clip.text_preprocessor, lang, os.path.join(datapath, f'{params.image_encoder}/{test_ds}.{lang}.pkl'), f'Tokenizing test {lang} with mclip') for lang in langs}
		
	train_text_embs, test_text_embs = {}, {}
	for lang in langs:
		embs_f = os.path.join(datapath, f'{params.image_encoder}/train.{lang}.pth')
		try:
			train_text_embs[lang] = torch.load(embs_f)
		except:
			text_ds = DocDataset(train_tok_mclip[lang])
			text_dl = DataLoader(text_ds, batch_size = 512, shuffle = False, num_workers = 4, pin_memory = True, collate_fn = collate_texts)
			train_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding train.{lang} mclip')

		embs_f = os.path.join(datapath, f'{params.image_encoder}/{test_ds}.{lang}.pth')
		try:
			test_text_embs[lang] = torch.load(embs_f)
		except:
			text_ds = DocDataset(test_tok_mclip[lang])
			text_dl = DataLoader(text_ds, batch_size = 512, shuffle = False, num_workers = 4, pin_memory = True, collate_fn = collate_texts)
			test_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding {test_ds}.{lang} mclip')

	return train_texts, test_texts, train_tok_mbart, test_tok_mbart, None, None, train_text_embs, test_text_embs, train_tok_mclip, test_tok_mclip