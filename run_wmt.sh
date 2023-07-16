#!/bin/tcsh -e
#SBATCH --job-name=wmt_en_fr_clip
#SBATCH --account=weidf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=150GB
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpua100
#SBATCH --exclusive
#SBATCH -e out/wmt_en_fr_clip.err

module load anaconda
conda activate cliptrans

# preprocessing
# python src/main.py --mn wmt --prefix_length 10 --bs 2 --update_count 8 --test_ds val --stage 3 --tgt_lang ro --ds wmt --epochs 6
# python src/main.py --mn wmt --prefix_length 10 --bs 2 --update_count 8 --test_ds test --stage 3 --test --tgt_lang ro --ds wmt --epochs 6

# en->ro
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt --prefix_length 10 --bs 4 --num_gpus 4 --update_count 8 --test_ds val --stage 4 --caption_ds multi30k --lm model_best_test.pth --ct --tgt_lang ro --ds wmt --epochs 15 > out/wmt_en_ro.out

# en->ro m-BART
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt_mbart --prefix_length 0 --bs 8 --num_gpus 4 --update_count 8 --test_ds val --stage 3 --tgt_lang ro --ds wmt --epochs 15 > out/wmt_en_ro_mbart.out

# Testing
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt_mbart --prefix_length 0 --bs 8 --num_gpus 4 --test_ds test --lm model_best_test.pth --test --stage 3 --tgt_lang ro --ds wmt >> out/wmt_en_ro_mbart.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt --prefix_length 10 --bs 8 --num_gpus 4 --test_ds test --stage 3 --lm model_best_test.pth --test --tgt_lang ro --ds wmt >> out/wmt_en_ro.out
python src/main.py --mn wmt_clip --prefix_length 10 --bs 4 --num_gpus 1 --update_count 16 --test_ds test --stage translate --tgt_lang fr --ds wmt --epochs 15 --image_encoder clip --preprocess_only
python src/main.py --mn wmt_clip --prefix_length 10 --bs 4 --num_gpus 1 --update_count 16 --test_ds val --stage translate --tgt_lang fr --ds wmt --epochs 15 --image_encoder clip --preprocess_only
python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt_clip --prefix_length 10 --bs 4 --num_gpus 4 --update_count 16 --test_ds val --stage translate --tgt_lang fr --ds wmt --epochs 15 --image_encoder clip > out/wmt_en_fr_clip.out
python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt_clip --prefix_length 10 --test --lm model_best_test.pth --bs 16 --num_gpus 4 --update_count 4 --test_ds test --stage translate --tgt_lang fr --ds wmt --epochs 15 --image_encoder clip --lm model_best_test.pth >> out/wmt_en_fr_clip.out

# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt_mbart --prefix_length 0 --bs 4 --num_gpus 4 --update_count 16 --test_ds val --stage translate --tgt_lang de --ds wmt --epochs 15 > out/wmt_en_de_mbart.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt_mbart --prefix_length 0 --test --lm model_best_test.pth --bs 16 --num_gpus 4 --update_count 4 --test_ds test --stage translate --tgt_lang de --ds wmt --epochs 15 >> out/wmt_en_de_mbart.out

# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt --prefix_length 10 --bs 8 --num_gpus 4 --update_count 8 --test_ds val --stage translate --tgt_lang de --ds wmt --epochs 15 --lm multi30k_fin-en-de/model_pretrained.pth > out/wmt_en_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn wmt --prefix_length 10 --test --lm model_best_test.pth --bs 16 --num_gpus 4 --update_count 4 --test_ds test --stage translate --tgt_lang de --ds wmt --epochs 15 >> out/wmt_en_de.out
