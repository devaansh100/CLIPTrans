import sys
sys.path.append('src/models')
sys.path.append('src/utils')

import torch
import argparse
import os
from multi30k import *
from wit import *
from wmt import *
from coco_captions import *
from model_utils import get_lang_code
from dataset import *
from model import CLIPTrans, CLIPTrans_CLIP
from runner import Runner
from torch.utils.data.distributed import DistributedSampler
from ddp import *
import numpy as np
import random
from collate_fns import *
import torch.nn as nn

get_ds = {'multi30k': get_Multi30k, 'wit': get_WIT, 'wmt': get_WMT, 'coco': get_coco}

def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(params):
    init_seed(params.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    params.model_name = os.path.join(params.model_dir, f'{params.model_name}-{params.src_lang}-{params.tgt_lang}')
    os.makedirs(params.model_name, exist_ok = True)
    MODEL = CLIPTrans if not params.unfreeze_clip else CLIPTrans_CLIP
    model = MODEL(params)
    if params.num_gpus > 1:
        init_distributed()
    else:
        torch.cuda.set_device(params.local_rank)
    model.cuda()
    tokenizer = model.tokenizer
    train_texts, test_texts, train_tok, test_tok, train_image_embs, test_image_embs, train_text_embs, test_text_embs, train_tok_mclip, test_tok_mclip = get_ds[params.ds](params, model, params.test_ds)
    if params.preprocess_only:
        exit()
    train_dataset_inputs = {'params': params, 'tok_data': train_tok, 'raw_data': train_texts, 'clip_tok_data': train_tok_mclip}
    test_dataset_inputs = {'params': params, 'tok_data': test_tok, 'raw_data': test_texts, 'clip_tok_data': test_tok_mclip}
    stage_message = {
                        'caption': 'Pretraining on Image Captions', 
                        'text_recon': 'Pretraining on Text Reconstruction', 
                        'translate': 'Pairwise Translation',
                        'triplet': 'Triplet Training'
                    }
    if is_main_process():
        print(stage_message[params.stage])

    train_dataset_inputs['is_pretraining'] = params.stage in ['caption', 'text_recon']
    test_dataset_inputs['is_pretraining'] = params.stage in ['caption', 'text_recon']

    train_dataset_inputs['mask_inputs'] = train_dataset_inputs['is_pretraining'] or params.noise_train
    test_dataset_inputs['mask_inputs'] = test_dataset_inputs['is_pretraining'] or params.noise_test

    if params.stage in ['text_recon', 'translate']:
        train_dataset_inputs['clip_embs'] = train_text_embs
        test_dataset_inputs['clip_embs'] = test_text_embs if not params.noise_test else test_image_embs
    
    elif params.stage in ['caption']:
        train_dataset_inputs['clip_embs'] = train_image_embs
        test_dataset_inputs['clip_embs'] = test_image_embs

    elif params.stage in ['triplet']:
        train_dataset_inputs['clip_embs'] = train_image_embs
        test_dataset_inputs['clip_embs'] = test_text_embs

    train_dataset = MultiModalDataset(**train_dataset_inputs)
    test_dataset = MultiModalDataset(**test_dataset_inputs)

    # if params.single_stage:
    #     # if params.caption_ds == '':
    #     #     if params.ds == 'wmt':
    #     #         params.caption_ds = 'multi30k'
    #     #     else:
    #     #         params.caption_ds = params.ds
    #     # if params.caption_ds != params.ds:
    #     #     train_texts, test_texts, train_tok, test_tok, train_image_embs, test_image_embs, train_text_embs, test_text_embs = get_ds[params.caption_ds](params, model, params.test_ds, force_pretraining = True)
    #     train_dataset_inputs['clip_embs'] = train_image_embs
    #     mix_dataset = MultiModalDataset(**train_dataset_inputs) # 'translate' -> captioning, 'triplet' -> text_recon
    #     train_dataset = ConcatDataset(train_dataset, mix_dataset)

    if not params.unfreeze_clip:
        del model.clip # If CLIP is always frozen, we can remove it from memory since all the data is preprocessed
        if is_main_process():
            print('Also finetuning CLIP.')
    if params.num_gpus > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        model = nn.parallel.DistributedDataParallel(model, device_ids = [local_rank], output_device = [local_rank], find_unused_parameters=True)
    if params.num_gpus > 1:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
        train_dl = DataLoader(train_dataset, batch_size = params.batch_size,
                          num_workers = 0, pin_memory = True, collate_fn = collate_multi, sampler=train_sampler)
        test_dl = DataLoader(test_dataset, batch_size = 2*params.batch_size,
                        num_workers = 6, pin_memory = True, collate_fn = collate_multi, sampler=test_sampler)
    else:
        train_dl = DataLoader(train_dataset, batch_size = params.batch_size, shuffle = True, 
                            num_workers = 0, pin_memory = True, collate_fn = collate_multi)
        test_dl = DataLoader(test_dataset, batch_size = params.batch_size, shuffle = False,
                            num_workers = 0, pin_memory = True, collate_fn = collate_multi)
    if is_main_process():
        if is_dist_avail_and_initialized():
            print(model.module.adapter) if hasattr(model.module, 'adapter') else print('No adapter')
        else:
            print(model.adapter) if hasattr(model, 'adapter') else print('No adapter')
        print('%' * 80)
        print(params)
        print('%' * 80)
    runner = Runner(train_dl, test_dl, params)
    runner.train(model, tokenizer, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mn', dest = 'model_name', type = str, default = '', help = 'Name of the job')
    parser.add_argument('--lm', dest = 'load_model', type = str, default = '', help = 'Name of model to be loaded')
    parser.add_argument('--test', action = 'store_true', help = 'to run inference on a saved model')
    parser.add_argument('--ct', dest = 'continue_training', action = 'store_true', help = 'flag to continue training')
    parser.add_argument('--bs', dest = 'batch_size', type = int, default = 128)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--epochs', type = int, default = 15)
    parser.add_argument('--model_dir', type = str, default = 'models')
    parser.add_argument('--data_dir', type = str, default = 'data')
    parser.add_argument('--src_lang', type = str, default = 'en')
    parser.add_argument('--tgt_lang', type = str, default = 'de')
    parser.add_argument('--update_count', type = int, default = 4, help = 'number of steps to accumulate gradient before backpropagating')
    parser.add_argument('--local_rank', type = int, default = 0, help = "Don't modify, will be used automatically for DDP")
    parser.add_argument('--num_gpus', type = int, default = 1)
    parser.add_argument('--test_after', type = int, default = -1)
    parser.add_argument('--seed', type = int, default = 29)
    parser.add_argument('--ds', type = str, choices = ['multi30k', 'wit', 'wmt', 'coco'], default = 'multi30k')
    parser.add_argument('--unfreeze_clip', action = 'store_true', help = 'used to also finetune the CLIP text encoder in stage 2')
    # parser.add_argument('--single_stage', action = 'store_true')
    # parser.add_argument('--caption_ds', type = str, choices = ['multi30k', 'wit', 'coco'], default = '')
    parser.add_argument('--test_ds', nargs = '+', type = str, default = ['2016', 'val'])
    parser.add_argument('--mapping_network', type = str, default = 'mlp', choices = ['mlp', 'transformer', 'hidden_mlp'], help = 'Choice of mapping network, refer paper for details')
    parser.add_argument('--image_encoder', type = str, default = 'mclip', choices = ['clip_res', 'mclip', 'clip'])
    parser.add_argument('--prefix_length', type = int, default = 10)
    parser.add_argument('--hidden_dims', type = int, default = 300)
    parser.add_argument('--mask_prob', type = float, default = 1)
    # parser.add_argument('--fusion', type = str, choices = ['context', 'gated_fusion'], default = 'context', help = 'Choose the fusion strategy')
    parser.add_argument('--stage', type = str, required = True, choices = ['caption', 'text_recon', 'translate', 'triplet'])
    parser.add_argument('--use_all_langs', action = 'store_true', help = 'Multilingual captioning in stage 1')
    parser.add_argument('--noise_train', action = 'store_true', help = 'Remove mask_prob% of the tokens while training')
    parser.add_argument('--noise_test', action = 'store_true', help = 'Remove mask_prob% of the tokens while testing')
    parser.add_argument('--preprocess_only', action = 'store_true')
    params = parser.parse_args()
    assert not (params.stage in ['caption', 'text_recon'] and params.ds == 'wmt'), 'While using text-only NMT, you cannot train stage 1. Make sure you load a stage 1 pretrained model'
    main(params)
