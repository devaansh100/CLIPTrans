# CLIPTrans
This repository is the official code implementation for the paper "Transferring Visual Knowledge with Pre-trained Models for Multimodal Machine Translation", accepted at ICCV'23.

# Setup

Environment details coming soon.

# Data

All data should be organised in the ```data/``` directory. Specific details for each dataset coming soon.

## Multi30k

## WIT

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

If the code and/or method was useful to your work, please consider citing us!
