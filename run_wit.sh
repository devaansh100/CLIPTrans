#!/bin/tcsh -e
#SBATCH --job-name=wit_unfrozen_clip_en_ro
#SBATCH --account=weidf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=250GB
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpua100
#SBATCH --exclusive
#SBATCH -e out/wit_unfrozen_clip_en_ro.err

module load anaconda
conda activate cliptrans
# # en -> ro
python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_unfrozen_clip --prefix_length 10 --bs 4 --num_gpus 4 --update_count 16 --lr 1e-5 --test_ds valid --stage translate --lm wit-en-de/model_pretrained.pth --tgt_lang ro --ds wit --unfreeze_clip > out/wit_unfrozen_clip_en_ro.out
python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_unfrozen_clip --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds test --stage translate --lm model_best_test.pth --tgt_lang ro --ds wit --unfreeze_clip --test >> out/wit_unfrozen_clip_en_ro.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_clip --prefix_length 10 --bs 16 --num_gpus 4 --update_count 4 --lr 1e-5 --test_ds valid --stage caption --tgt_lang ro --ds wit --image_encoder clip >> out/wit_clip_en_ro.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_clip --prefix_length 10 --bs 16 --num_gpus 4 --update_count 4 --lr 1e-5 --test_ds valid --stage translate --lm model_pretrained.pth --tgt_lang ro --ds wit --image_encoder clip >> out/wit_clip_en_ro.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_clip --prefix_length 10 --bs 16 --num_gpus 4 --test_ds test --test --stage translate --lm model_best_test.pth --src_lang en --tgt_lang ro --ds wit --image_encoder clip >> out/wit_clip_en_ro.out

# #  en -> af
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_clip --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage caption --tgt_lang af --ds wit --image_encoder clip > out/wit_clip_en_af.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_clip --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --lm model_pretrained.pth --tgt_lang af --ds wit --image_encoder clip >> out/wit_clip_en_af.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_clip --prefix_length 10 --bs 16 --num_gpus 4 --test_ds test --test --stage translate --lm model_best_test.pth --src_lang en --tgt_lang af --ds wit --image_encoder clip >> out/wit_clip_en_af.out

# # de -> es
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_fr --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage caption --src_lang de --tgt_lang fr --ds wit > out/wit_de_fr_2.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_fr --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --src_lang de --tgt_lang fr --ds wit --lm model_pretrained.pth >> out/wit_de_fr_2.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_fr --prefix_length 10 --bs 16 --num_gpus 4 --test_ds test --test --stage translate --lm model_best_test.pth --src_lang de --tgt_lang fr --ds wit >> out/wit_de_fr_2.out

# # es -> fr
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage caption --src_lang es --tgt_lang fr --ds wit > out/wit_fr_fr_2.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --src_lang es --tgt_lang fr --ds wit --lm model_pretrained.pth  >> out/wit_fr_fr_2.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 10 --bs 16 --num_gpus 4 --test_ds test --test --stage translate --lm model_best_test.pth --src_lang es --tgt_lang fr --ds wit >> out/wit_fr_fr_2.out

# # en -> fr
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 10 --bs 16 --num_gpus 4 --update_count 4 --lr 1e-5 --test_ds valid --stage caption --tgt_lang fr --ds wit > out/wit_en_fr_ep_10.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --lm model_pretrained.pth --tgt_lang fr --ds wit  >> out/wit_en_fr_ep_10.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 10 --bs 16 --num_gpus 4 --test_ds test --test --stage translate --lm model_best_test.pth --src_lang en --tgt_lang fr --ds wit >> out/wit_en_fr_ep_10.out

# # en -> de
# python src/main.py --mn wit --prefix_length 10 --bs 16 --update_count 4 --lr 1e-5 --test_ds valid --stage caption --src_lang en --tgt_lang de --ds wit --preprocess_only
# python src/main.py --mn wit --prefix_length 10 --bs 16 --update_count 4 --lr 1e-5 --test_ds test --stage caption --src_lang en --tgt_lang de --ds wit --preprocess_only

# en -> es
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 10 --bs 4 --num_gpus 4 --update_count 16 --lr 1e-5 --test_ds valid --stage caption --tgt_lang es --ds wit --epochs 15 > out/wit_en_es.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --epochs 15 --mn wit --prefix_length 10 --bs 4 --num_gpus 4 --update_count 16 --lr 1e-5 --test_ds valid --stage translate --lm model_pretrained.pth --tgt_lang es --ds wit >> out/wit_en_es.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 10 --bs 4 --num_gpus 4 --test_ds test --test --stage translate --lm model_best_test.pth --src_lang en --tgt_lang es --ds wit >> out/wit_en_es.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --epochs 15 --mn wit --prefix_length 0 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --tgt_lang fr --ds wit >> out/wit_en_fr_mbart.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit --prefix_length 0 --bs 4 --num_gpus 4 --test_ds test --test --stage translate --lm model_best_test.pth --src_lang en --tgt_lang fr --ds wit >> out/wit_en_fr_mbart.out

# # en -> ro m-BART
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_mbart --prefix_length 0 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --tgt_lang ro --ds wit > out/wit_en_ro_mbart.out

# # en -> af m-BART
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_mbart --prefix_length 0 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --tgt_lang af --ds wit > out/wit_en_af_all_langs_mbart.out

# # de -> es m-BART
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_mbart --prefix_length 0 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --src_lang de --tgt_lang fr --ds wit > out/wit_de_fr_mbart.out

# # es -> fr m-BART
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wit_mbart --prefix_length 0 --bs 8 --num_gpus 4 --update_count 8 --lr 1e-5 --test_ds valid --stage translate --src_lang es --tgt_lang fr --ds wit > out/wit_fr_fr_mbart.out
