import torch

def freeze_params(module):
	for param in module.parameters():
		param.requires_grad = False

def get_lang_code(lang):
	LANG_MAP = {
				'ar': 'ar_AR', 'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 'et': 'et_EE', 
				'fi': 'fi_FI', 'fr': 'fr_XX', 'gu': 'gu_IN', 'hi': 'hi_IN', 'it': 'it_IT', 'ja': 'ja_XX', 
				'kk': 'kk_KZ', 'ko': 'ko_KR', 'lt': 'lt_LT', 'lv': 'lv_LV', 'my': 'my_MM', 'ne': 'ne_NP', 
				'nl': 'nl_XX', 'ro': 'ro_RO', 'ru': 'ru_RU', 'si': 'si_LK', 'tr': 'tr_TR', 'vi': 'vi_VN', 
				'zh': 'zh_CN', 'af': 'af_ZA'
			}
	return LANG_MAP[lang]

def send_to_cuda(batch):
	if torch.is_tensor(batch):
		batch = batch.cuda()
	else:
		for key in batch:
			if torch.is_tensor(batch[key]):
				batch[key] = batch[key].cuda()
	return batch