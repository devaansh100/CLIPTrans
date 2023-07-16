import sys
sys.path.append('utils')
import torch
from tqdm import tqdm
import sacrebleu
import torch.optim as optim
from torch_poly_lr_decay import PolynomialLRDecay
from ddp import *
from model_utils import send_to_cuda
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import math
from model_utils import get_lang_code
import warnings
import evaluate

class Runner:
    def __init__(self, train_dl, test_dl, params):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.update_count = params.update_count
        self.test_after = params.test_after * self.update_count
        self.is_pretraining = params.stage in ['caption', 'text_recon']

    def save_model(self, model, name, epoch):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch,
            'best_bleu_test': self.best_bleu_test
        }
        torch.save(checkpoint, name)

    def load_model(self, params, model, name, load_opt):
        checkpoint = torch.load(name, map_location = torch.device('cpu'))
        if params.num_gpus == 1:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            checkpoint['model'] = new_state_dict
        model.load_state_dict(checkpoint['model'], strict = False)
        
        if load_opt:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'], strict = False)
                self.scaler.load_state_dict(checkpoint['scaler'], strict = False)
            except:
                warnings.warn('Could not load optimizer due to extra parameters')
        else:
            warnings.warn('Not loading optimizer - if you intended to continue training(and not just load weights), hard code load_opt = True and rerun')
        model.cuda()

        return checkpoint['epoch'], checkpoint['best_bleu_test']

    def fit_one_epoch(self, model, tokenizer, params, epoch):
        model.train()
        train_loss = 0.0
        prob = torch.rand(1).item()
        self.optimizer.zero_grad()
        for step, batch in enumerate(tqdm(self.train_dl, desc = f'Epoch {epoch}', disable = not is_main_process())):
            batch['mbart'], batch['clip'] = send_to_cuda(batch['mbart']), send_to_cuda(batch['clip'])
            with torch.autocast(device_type='cuda'):
                output = model(batch)
                loss = output[0]
            self.scaler.scale(loss).backward()
            if (step + 1) % self.update_count == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.cycle_scheduler.step()
                self.optimizer.zero_grad()
            if params.num_gpus > 1:
                loss_collated = [torch.zeros_like(loss).cuda() for _ in range(params.num_gpus)]
                dist.all_gather(loss_collated, loss)
                train_loss = sum(loss_collated).item()
            else:
                train_loss = loss.item()
            del batch
            if self.test_after > 0 and (step + 1) % self.test_after == 0:
                self.test(model, tokenizer, params, epoch)
        
        if is_main_process():
            print(f'Epoch {epoch}: Train Loss: {self.update_count * train_loss/(params.num_gpus * len(self.train_dl))}\n')
            if self.is_pretraining:
                self.save_model(model, f'{params.model_name}/model_pretrained.pth', epoch)

    def test(self, model, tokenizer, params, epoch):
        model.eval()
        test_loss = 0.0
        translated_sentences, target_sentences = [], []
        tokenizer.tgt_lang = get_lang_code(params.tgt_lang)
        # meteor = evaluate.load('meteor')
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_dl, desc = f'Epoch {epoch}', disable = not is_main_process())):
                batch['clip'] = send_to_cuda(batch['clip'])
                batch['mbart'] = send_to_cuda(batch['mbart'])
                raw_target_text = batch.pop('raw')
                with torch.autocast(device_type='cuda'):
                    output = model(batch, mode = 'test')
                output = tokenizer.batch_decode(output, skip_special_tokens = True)
                if params.num_gpus > 1:
                    output_collated = [None for _ in range(params.num_gpus)]
                    dist.all_gather_object(output_collated, output)

                    targets_collated = [None for _ in range(params.num_gpus)]
                    dist.all_gather_object(targets_collated, raw_target_text)
                if is_main_process():
                    if params.num_gpus > 1:
                        for gpu_list in output_collated:
                            translated_sentences.extend(gpu_list)  
                        for gpu_list in targets_collated:
                            target_sentences.extend(gpu_list)        
                    else:
                        translated_sentences.extend(output)
                        target_sentences.extend(raw_target_text)

        if is_main_process():
            bleu_score = sacrebleu.corpus_bleu(translated_sentences, [target_sentences]).score
            meteor_score = 0 # meteor.compute(predictions = translated_sentences, references = target_sentences)['meteor']
            print(f'Epoch {epoch}; Test BLEU: {bleu_score}; Test METEOR: {100 * meteor_score}')
            print('------------------------------------------')
            for i, (tra, tgt) in enumerate(zip(translated_sentences[0:5], target_sentences[0:5])):
                print(f'Target Sentence {i}: {tgt}')
                print(f'Translated Sentence {i}: {tra}')
                print('------------------------------------------')

            if bleu_score > self.best_bleu_test and not self.is_pretraining and not params.test:
                self.best_bleu_test = bleu_score
                self.save_model(model, f'{params.model_name}/model_best_test.pth', epoch)

    def train(self, model, tokenizer, params):
        self.best_bleu_test = -float('Inf')
        self.optimizer = optim.AdamW(model.parameters(), lr = params.lr, betas = (0.9, 0.98), eps = 1e-6, weight_decay = 0.0)
        self.scaler = GradScaler()
        steps_per_epoch = math.ceil(len(self.train_dl)/self.update_count)
        last_epoch, last_batch = 0, -1
        self.cycle_scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps = 40000, end_learning_rate = params.lr, power = 2.0)
        if params.load_model:
            last_epoch, self.best_bleu_test = self.load_model(params, model, f'{params.model_name}/{params.load_model}' if '/' not in params.load_model else f'{params.model_dir}/{params.load_model}', load_opt = True)
            if params.continue_training:
                last_batch = (last_epoch - 1) * steps_per_epoch
                for step in range(last_batch):
                    self.cycle_scheduler.step()
            elif not params.test: # Load a model but restart training
                last_epoch, last_batch = 0, -1 
                self.best_bleu_test = -float('Inf')
        if params.test:
            self.test(model, tokenizer, params, last_epoch)
            return
        for epoch in range(last_epoch, params.epochs):
            if params.num_gpus > 1:
                self.train_dl.sampler.set_epoch(epoch)
                self.test_dl.sampler.set_epoch(epoch)
            self.fit_one_epoch(model, tokenizer, params, epoch+1)
            self.test(model, tokenizer, params, epoch+1)
