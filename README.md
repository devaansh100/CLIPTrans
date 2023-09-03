# CLIPTrans
[[Project page]](https://devaansh100.github.io/projects/cliptrans/) [[arxiv]](https://arxiv.org/abs/2308.15226)
Official implementation for the paper "CLIPTrans: Transferring Visual Knowledge with Pre-trained Models for Multimodal Machine Translation", published at ICCV'23. The aim of the paper is to leverage existing pre-trained models(multilingual mBART and multimodal M-CLIP) for multimodal machine translation. More generally, it proposes a framework(pictured below) for multilingual generative tasks using multimodal data.

![Training Pipeline GIF](media/training.gif)

# Setup

Setup the repository with the following commands:
```bash
git clone --recursive git@github.com:devaansh100/CLIPTrans.git
conda create cliptrans -f environment.yml
conda activate cliptrans
pip install -r requirements.txt
cd transformers
pip install -e .
```

# Data

All data should be organised in the ```data/``` directory.

## Multi30k 

Download the images for [Flickr30k](https://github.com/multi30k/dataset/tree/master#visual-features) and the [MSCOCO test set](https://www.statmt.org/wmt17/multimodal-task.html) in the ```data/multi30k``` folder. You should have three tarballs for the images: ```flickr30k-images.tar.gz``` which will have the training, validation and Test2016 set; ```test_2017-flickr-images.gz``` for the MSCOCO set and ```images_mscoco.task1.tar.gz``` for the MSCOCO images.

Now, simply execute ```setup_multi30k.sh``` to download the text data and organise the folders.

## WIT

The WIT dataset used in our paper can be downloaded from [here](https://github.com/JerryYLi/valhalla-nmt/releases) into the ```data/wit``` folder. Once downloaded, unzip with ```tar -xvzf wit_mmt.tar.gz```. The images for each of these can be downloaded by running the following command:
```bash
python download_images.py $FOLDER_NAME $SPLIT
```
where ```FOLDER_NAME``` can be one of [```es_fr```, ```en_ro```, ```en_fr```, ```en_es```, ```en_de```, ```en_af```, ```de_es```] and ```SPLIT``` can be one of [```train```, ```test```, ```valid```]. This will take a while. Also note that the downloading of images depends upon the availability of the image on the hosted service; due to which some variance in the scores is to be expected.

# Training

Training is done in two stages. To run the first stage(captioning), the following commands can be used, depending on the number of available GPUs:
```bash
python src/main.py --num_gpus 1 --mn multi30k --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2016 val --stage caption --tgt_lang fr
```

```bash
python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2016 val --stage caption --tgt_lang fr
```

For stage 2, use the following commands:
```bash
python src/main.py --num_gpus 1 --mn multi30k --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2016 val --stage translate --tgt_lang fr --lm model_pretrained.pth
```

```bash
python -m torch.distributed.run --nproc_per_node 4 src/main.py --num_gpus 4 --mn multi30k --prefix_length 10 --bs 32 --update_count 4 --lr 1e-5 --test_ds 2016 val --stage translate --tgt_lang fr --lm model_pretrained.pth
```

Here is a quick guide to some specifics about the flags:
1. ```--stage``` denotes the training task. There are four choices available which are detailed in the table below. These affect the training task, and inference is modified appropriately:

| --stage    | CLIP input  | mBART input | mBART output  |
|------------|-------------|-------------|---------------|
| caption    | image       | trivial     | image caption |
| translate  | source text | source text | target text   |
| text_recon | source text | trivial     | source text   |
| triplet    | image       | source text | target text   |

2. ```--mn``` sets the name of the job. It is used to create a unique model folder where the weights are stored and can be loaded from. The source and target language are appended to this name. It must remain uniform across stage 1 and stage 2.
3. ```--lm``` is the name of the weights file to be loaded(which gets saved in the aforementioned folder). For final results, load ```model_best_test.pth```. Stage 1 models are saved as ```model_pretrained.pth```. To continue training from a saved model, load its weights and add the flag ```--ct```.
4. ```--test_ds``` sets the dataset to be used in validation/test. While training, pass ```2016 val```(Multi30k) or ```valid```(WIT). For inference, pass ```2016 flickr```, ```2017 flickr``` or ```2017 mscoco``` for Multi30k and ```test``` for WIT. Also add the flag ```--test``` so that only inference is run for a saved model.
5. To finetune an mBART on a dataset, simply pass ```--prefix_length 0```.
6. To use images in inference, add the flags ```--noise_test --mask_prob 0``` to the inference command.

If the code and/or method was useful to your work, please consider citing us!
```code
@inproceedings {gupta2023cliptrans,
    title={CLIPTrans: Transferring Visual Knowledge with Pre-trained Models for Multimodal Machine Translation},
    author={Gupta, Devaansh and Kharbanda, Siddhant and Zhou, Jiawei and Li, Wanhua and Pfister, Hanspeter and Wei, Donglai},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    year={2023}
}```

