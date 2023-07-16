import torch
from tqdm import tqdm
import pickle as pkl
from PIL import Image
import os
from model_utils import get_lang_code, send_to_cuda
from collate_fns import *
from dataset import DocDataset
from torch.utils.data import DataLoader

def get_image_embs(clip, folder, image_splits, image_embs_f, desc, preprocessor, ignore_indices=[]):
	try:
		img_embs = torch.load(image_embs_f)
	except:
		img_paths = [os.path.join(folder, f) for f in image_splits]
		imgs = []
		for i, path in enumerate(tqdm(img_paths, desc = 'Reading images')):
			if i in ignore_indices:
				continue
			if '#' in path:
				path = path[:path.index('#')] # For multi30k mscoco test splits
			try:
				img = read_image(path, preprocessor)
				imgs.append(img)
			except:
				ignore_indices.append(i)
		img_ds = DocDataset(imgs)
		img_dl = DataLoader(img_ds, batch_size = 128, shuffle = False, num_workers = 4, pin_memory = True, collate_fn = collate_images)
		img_embs = create_embeddings(img_dl, clip, image_embs_f, desc)
	return img_embs

def read_image(img_path, preprocessor):
	img = Image.open(img_path)
	try:
		img = torch.from_numpy(preprocessor(images=img)['pixel_values'][0]) # For compatibility with Huggingface
	except:
		img = preprocessor(img) # For compatibility with openai clip

	return img

def tokenize(texts, tokenizer, lang, outfile, desc = ''):
	if not os.path.exists(outfile):
		data = []
		tokenizer.src_lang = get_lang_code(lang)
		for text in tqdm(texts, desc = desc):
			data.append(tokenizer(text, return_tensors = 'pt', truncation = True))
		
		with open(outfile, 'wb') as f:
			pkl.dump(data, f)
	else:
		with open(outfile, 'rb') as f:
			data = pkl.load(f)
	return data

def create_embeddings(dl, encoder, outfile, desc = ''):
	embs = torch.tensor([]).cuda()
	with torch.no_grad():
		with torch.autocast(device_type='cuda'):
			for data in tqdm(dl, desc = desc):
				embs = torch.cat((embs, encoder(send_to_cuda(data))), dim = 0)
	torch.save(embs.cpu(), outfile)
	return embs

def postprocess_pairs(train_texts, test_texts, train_tok_mbart, test_tok_mbart, train_img_embs, test_img_embs, train_text_embs, test_text_embs, params, train_ignore_indices, test_ignore_indices, train_tok_mclip, test_tok_mclip, force_pretraining):
	if params.stage in ['caption', 'text_recon'] or force_pretraining: # Need paired images in this stage, use ignore_indices to remove the respective test indices. force_pretraining used for caption_dataset in stage 4
		for index in sorted(train_ignore_indices[::-1], reverse = True): # Reversing before sorting since most of the list is already sorted in ascending order -> faster sorting
			for lang in train_texts.keys():
				train_texts[lang].pop(index)
				train_tok_mbart[lang].pop(index)
		
		train_mask = torch.isin(torch.arange(train_text_embs[params.src_lang].shape[0]), torch.tensor(train_ignore_indices), invert = True)
		for lang in train_texts.keys():
			train_text_embs[lang] = train_text_embs[lang][train_mask]
		
		for index in sorted(test_ignore_indices[::-1], reverse = True): # Reversing before sorting since most of the list is already sorted in ascending order -> faster sorting
			for lang in test_texts.keys():
				test_texts[lang].pop(index)
				test_tok_mbart[lang].pop(index)
		
		test_mask = torch.isin(torch.arange(test_text_embs[params.src_lang].shape[0]), torch.tensor(test_ignore_indices), invert = True)
		for lang in test_texts.keys():
			test_text_embs[lang] = test_text_embs[lang][test_mask]
	
		# Stage 2 and 3 dont use image embs so they can use the complete text
		for lang in train_texts.keys():
			assert len(train_texts[lang]) == len(train_tok_mbart[lang]) == len(train_text_embs[lang]), 'Misalignment in train text pairs'
			if params.stage == 1:
				assert len(train_texts[lang]) == len(train_tok_mbart[lang]) == len(train_text_embs[lang]) == len(train_img_embs), 'Misalignment in train text pairs with images'
		for lang in test_texts.keys():
			assert len(test_texts[lang]) == len(test_tok_mbart[lang]) == len(test_text_embs[lang]), 'Misalignment in train text pairs'
			if params.stage == 1:
				assert len(test_texts[lang]) == len(test_tok_mbart[lang]) == len(test_text_embs[lang]) == len(test_img_embs), 'Misalignment in test text pairs with images'
		
	return train_texts, test_texts, train_tok_mbart, test_tok_mbart, train_img_embs, test_img_embs, train_text_embs, test_text_embs, train_tok_mclip, test_tok_mclip