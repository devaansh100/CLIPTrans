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

def get_Multi30k(params, model, test = ('2017', 'mscoco'), force_pretraining = False):
	if force_pretraining:
		langs = ['en'] # Only pretraining on en images
		test = ('2017', 'mscoco') # Anyway not going to be used
	else:
		langs =  [params.src_lang, params.tgt_lang]
	datapath = os.path.join(params.data_dir, 'multi30k')
	os.makedirs(os.path.join(datapath, f'text/data/task1/mbart'), exist_ok = True)
	os.makedirs(os.path.join(datapath, f'text/data/task1/{params.image_encoder}'), exist_ok = True)
	# Reading train files
	train_texts = {lang: open(os.path.join(datapath, f'text/data/task1/raw/train.{lang}')).read().splitlines() for lang in langs}
	try:
		train_tok_mbart = {lang: pkl.load(open(os.path.join(datapath, f'text/data/task1/mbart/train.{lang}.pkl'), 'rb')) for lang in langs}
	except:
		print('Did not find mbart train tokenized data. Creating...')
		train_tok_mbart = {lang: tokenize(train_texts[lang], model.tokenizer, lang, os.path.join(datapath, f'text/data/task1/mbart/train.{lang}.pkl'), f'Tokenizing train {lang} with mbart') for lang in langs}

	try:
		train_tok_mclip = {lang: pkl.load(open(os.path.join(datapath, f'text/data/task1/{params.image_encoder}/train.{lang}.pkl'), 'rb')) for lang in langs}
	except:
		print('Did not find mclip train tokenized data. Creating...')
		train_tok_mclip = {lang: tokenize(train_texts[lang], model.clip.text_preprocessor, lang, os.path.join(datapath, f'text/data/task1/{params.image_encoder}/train.{lang}.pkl'), f'Tokenizing train {lang} with {params.image_encoder}') for lang in langs}

	
	# Reading test files
	test_texts = {lang: open(os.path.join(datapath, f'text/data/task1/raw/test_{test[0]}_{test[1]}.{lang}')).read().splitlines() for lang in langs}
	try:
		test_tok_mbart = {lang: pkl.load(open(os.path.join(datapath, f'text/data/task1/mbart/test_{test[0]}_{test[1]}.{lang}.pkl'), 'rb')) for lang in langs}
	except:
		print('Did not find mbart test tokenized data. Creating...')
		test_tok_mbart = {lang: tokenize(test_texts[lang], model.tokenizer, lang, os.path.join(datapath, f'text/data/task1/mbart/test_{test[0]}_{test[1]}.{lang}.pkl'), f'Tokenizing test {lang} with mbart') for lang in langs}

	try:
		test_tok_mclip = {lang: pkl.load(open(os.path.join(datapath, f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pkl'), 'rb')) for lang in langs}
	except:
		print('Did not find mclip test tokenized data. Creating...')
		test_tok_mclip = {lang: tokenize(test_texts[lang], model.clip.text_preprocessor, lang, os.path.join(datapath, f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pkl'), f'Tokenizing test {lang} with {params.image_encoder}') for lang in langs}

	train_image_splits = open(os.path.join(datapath, f'text/data/task1/image_splits/train.txt')).read().splitlines()
	test_image_splits = open(os.path.join(datapath, f'text/data/task1/image_splits/test_{test[0]}_{test[1]}.txt')).read().splitlines()

	# Getting images and embedding with CLIP. Same for text
	print('Loaded all text files. Getting images...')
	train_img_embs = get_image_embs(model.clip, os.path.join(datapath, 'images/train'), train_image_splits, os.path.join(datapath, f'text/data/task1/{params.image_encoder}/train.pth'), 'Embedding train images', model.clip.image_preprocessor)
	test_img_embs = get_image_embs(model.clip, os.path.join(datapath, f'images/test_{test[0]}_{test[1]}'), test_image_splits, os.path.join(datapath, f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.pth'), f'Embedding test_{test[0]}_{test[1]} images', model.clip.image_preprocessor)
	
	train_text_embs, test_text_embs = {}, {}
	for lang in langs:
		embs_f = os.path.join(datapath, f'text/data/task1/{params.image_encoder}/train.{lang}.pth')
		try:
			train_text_embs[lang] = torch.load(embs_f)
		except:
			text_ds = DocDataset(train_tok_mclip[lang])
			text_dl = DataLoader(text_ds, batch_size = 256, shuffle = False, num_workers = 0, pin_memory = True, collate_fn = collate_texts)
			train_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding train.{lang} mclip')

		embs_f = os.path.join(datapath, f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pth')
		try:
			test_text_embs[lang] = torch.load(embs_f)
		except:
			text_ds = DocDataset(test_tok_mclip[lang])
			text_dl = DataLoader(text_ds, batch_size = 256, shuffle = False, num_workers = 4, pin_memory = True, collate_fn = collate_texts)
			test_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding test_{test[0]}_{test[1]}.{lang} mclip')

	return train_texts, test_texts, train_tok_mbart, test_tok_mbart, train_img_embs, test_img_embs, train_text_embs, test_text_embs, train_tok_mclip, test_tok_mclip
