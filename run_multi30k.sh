#!/bin/tcsh -e
#SBATCH --job-name=multi30k_en_cs_clip
#SBATCH --account=weidf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=150GB
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpua100
#SBATCH --exclusive
#SBATCH -e out/multi30k_en_cs_clip.err

module load anaconda
conda activate cliptrans

# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_unfrozen_clip --prefix_length 10 --bs 16 --update_count 4 --lr 1e-5 --test_ds 2016 val --stage translate --lm multi30k_fin-en-de/model_pretrained.pth --tgt_lang de --unfreeze_clip > out/multi30k_en_de_unfrozen_clip.out
python src/main.py --num_gpus 1 --mn multi30k_clip --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2016 val --stage caption --tgt_lang cs --preprocess_only --image_encoder clip
python src/main.py --num_gpus 1 --mn multi30k_clip --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2016 flickr --stage caption --tgt_lang cs --preprocess_only --image_encoder clip
python src/main.py --num_gpus 1 --mn multi30k_clip --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2018 flickr --stage caption --tgt_lang cs --preprocess_only --image_encoder clip
python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2016 val --stage caption --tgt_lang cs --image_encoder clip > out/multi30k_en_cs_clip.out
python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2016 val --stage translate --tgt_lang cs --image_encoder clip --lm model_pretrained.pth >> out/multi30k_en_cs_clip.out
python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip --prefix_length 10 --bs 32 --update_count 8 --lr 1e-5 --test_ds 2016 flickr --stage translate --image_encoder clip --lm model_best_test.pth --tgt_lang cs --test >> out/multi30k_en_cs_clip.out
python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip --prefix_length 10 --bs 32 --update_count 8 --lr 1e-5 --test_ds 2018 flickr --stage translate --image_encoder clip --lm model_best_test.pth --tgt_lang cs --test >> out/multi30k_en_cs_clip.out
# en -> cs
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_fin --prefix_length 10 --bs 32  --update_count 2 --lr 1e-5 --test_ds 2016 val --stage translate --lm multi30k_fin-en-de/model_pretrained.pth --tgt_lang cs > out/multi30k_fin_en_cs.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_fin --prefix_length 10 --bs 32  --update_count 2 --lr 1e-5 --test_ds 2016 flickr --stage translate --lm model_best_test.pth --tgt_lang cs --test >> out/multi30k_fin_en_cs.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_fin --prefix_length 10 --bs 32  --update_count 2 --lr 1e-5 --test_ds 2018 flickr --stage translate --lm model_best_test.pth --tgt_lang cs --test >> out/multi30k_fin_en_cs.out
# en -> de
# python src/main.py --num_gpus 1 --mn multi30k_clip_res --prefix_length 10 --bs 32 --update_count 2 --lr 1e-5 --test_ds 2017 mscoco --stage caption --image_encoder clip_res --preprocess_only > multi30k_clip_res_de.out
# python src/main.py --num_gpus 1 --mn multi30k_clip_res --prefix_length 10 --bs 32 --update_count 2 --lr 1e-5 --test_ds 2017 flickr --stage caption --image_encoder clip_res --preprocess_only >> multi30k_clip_res_de.out
# python src/main.py --num_gpus 1 --mn multi30k_clip_res --prefix_length 10 --bs 32 --update_count 2 --lr 1e-5 --test_ds 2016 flickr --stage caption --image_encoder clip_res --preprocess_only >> multi30k_clip_res_de.out
# python src/main.py --num_gpus 1 --mn multi30k_clip_res --prefix_length 10 --bs 32 --update_count 2 --lr 1e-5 --test_ds 2016 val --stage caption --image_encoder clip_res --preprocess_only >> multi30k_clip_res_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip_res --prefix_length 10 --bs 32 --update_count 2 --lr 1e-5 --test_ds 2016 val --stage caption --image_encoder clip >> out/multi30k_clip_res_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip_res --prefix_length 10 --bs 32 --update_count 2 --lr 1e-5 --test_ds 2016 val --stage translate --lm model_pretrained.pth --image_encoder clip >> out/multi30k_clip_res_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip_res --prefix_length 10 --bs 32 --test_ds 2016 flickr --stage translate --lm model_best_test.pth --test --image_encoder clip >> out/multi30k_clip_res_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip_res --prefix_length 10 --bs 32 --test_ds 2017 flickr --stage translate --lm model_best_test.pth --test --image_encoder clip >> out/multi30k_clip_res_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_clip_res --prefix_length 10 --bs 32 --test_ds 2017 mscoco --stage translate --lm model_best_test.pth --test --image_encoder clip >> out/multi30k_clip_res_de.out

# en -> fr
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_end --prefix_length 10 --bs 32 --update_count 2 --lr 1e-5 --test_ds 2016 val --stage caption --tgt_lang fr > out/multi30k_end_fr.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_end --prefix_length 10 --bs 32 --update_count 2 --lr 1e-5 --test_ds 2016 val --stage translate --lm model_pretrained.pth --tgt_lang fr >> out/multi30k_end_fr.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_end --prefix_length 10 --bs 32 --test_ds 2016 flickr --stage translate --lm model_best_test.pth --test --tgt_lang fr >> out/multi30k_end_fr.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_end --prefix_length 10 --bs 32 --test_ds 2017 flickr --stage translate --lm model_best_test.pth --test --tgt_lang fr >> out/multi30k_end_fr.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k_end --prefix_length 10 --bs 32 --test_ds 2017 mscoco --stage translate --lm model_best_test.pth --test --tgt_lang fr >> out/multi30k_end_fr.out

# en -> de mbart
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn multi30k_end_mbart --prefix_length 0 --bs 32 --num_gpus 4 --update_count 2 --lr 1e-5 --test_ds 2016 val --stage translate --noise_train --mask_prob 0.2 > out/multi30k_end_mbart_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn multi30k_end_mbart --prefix_length 0 --bs 32 --num_gpus 4 --test_ds 2016 flickr --stage translate --lm model_best_test.pth --test >> out/multi30k_end_mbart_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn multi30k_end_mbart --prefix_length 0 --bs 32 --num_gpus 4 --test_ds 2017 flickr --stage translate --lm model_best_test.pth --test >> out/multi30k_end_mbart_de.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn multi30k_end_mbart --prefix_length 0 --bs 32 --num_gpus 4 --test_ds 2017 mscoco --stage translate --lm model_best_test.pth --test >> out/multi30k_end_mbart_de.out

# en -> fr mbart
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn multi30k_end_mbart --prefix_length 0 --bs 32 --num_gpus 4 --update_count 2 --lr 1e-5 --test_ds 2016 val --stage translate --tgt_lang fr --noise_train --mask_prob 0.8 > out/multi30k_end_mbart_fr.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn multi30k_end_mbart --prefix_length 0 --bs 32 --num_gpus 4 --test_ds 2016 flickr --stage translate --tgt_lang fr --lm model_best_test.pth --test >> out/multi30k_end_mbart_fr.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn multi30k_end_mbart --prefix_length 0 --bs 32 --num_gpus 4 --test_ds 2017 flickr --stage translate --tgt_lang fr --lm model_best_test.pth --test >> out/multi30k_end_mbart_fr.out
# python -m torch.distributed.run --nproc_per_node 4 src/main.py --mn multi30k_end_mbart --prefix_length 0 --bs 32 --num_gpus 4 --test_ds 2017 mscoco --stage translate --tgt_lang fr --lm model_best_test.pth --test >> out/multi30k_end_mbart_fr.out
