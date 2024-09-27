# Image-Supervised Generative Models for Vector Graphics
Code for the master's thesis "Image-Supervised Generative Models for Vector Graphics" by Moritz Feuerpfeil. The prepared scripts can train, eval, and sample for the proposed architecture and all baseline comparison models. 

The proposed adapted VQ-VAE is split into `models/vsq.py` ($VSQ$) and `models/stage2.py` ($TM$). You can find the VQTokenizer in `tokenizer.py`, which uses the VSQ to prepare the data for the TM model. If not specified, then all scripts assume your cwd is the top-level directory of this repository.

If you have questions or need more scripts to re-do some of the sweeps or experiments, feel free to open an issue.

# Installation
## Environment
```bash
conda env create -f conda_env.yaml
conda activate thesis
```

DiffVG has to be installed separately.
```bash
git clone git@github.com:BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```

## Data
All data will be stored in the `.data` directory. This whole section is only required if you want to evaluate or train a model.

### Split from IconShop
We will need some data from the IconShop data splits, so head over to [the Google drive they published](https://drive.google.com/drive/folders/1dXdrBqJDmEn8K8TeY2w3mvEtADZipPOc) and download `FIGR-SVG-train.csv`, `FIGR-SVG-test.csv`, `FIGR-SVG-valid.csv` and place them in `.data/`.

### FIGR-8
Please download the FIGR-8 dataset: https://github.com/marcdemers/FIGR-8 and place it in `.data/` so that the final path looks like this: `.data/FIGR-8/Data/`.

this should do it (you need git lfs installed):
```bash
cd .data
git clone git@github.com:marcdemers/FIGR-8.git
cd FIGR-8
git lfs pull
```
(this will take some time as there are 1.5M files)

### Make the data split & intermediate vector format
```bash
conda activate thesis
bash ./make_data.sh
```

# Train
The nicely prepared part of this repository lets you train a $TM_l$ model on the full data split. This requires a $VSQ_l$ model, which is trained first. If you want to make any modifications, feel free to edit the script.
```bash
bash train.sh
```

# Evaluate
In general, all these scripts assume your GPU is available. If you have settings that require an explicit CUDA_VISIBLE_DEVICES env variable set, please adjust the shell scripts.
## $TM$ models
You can evaluate the self-trained checkpoint or the one I provide that was used in the thesis.

If you want to use my pre-trained checkpoints, please download the directories `VSQ_l` and `TM_l` from [this Google drive](https://drive.google.com/drive/folders/1kEHkt0RVzHw7nwYADqAktYHlsVlWoFi7?usp=sharing) and place it in `results`, so that the two folders `results/VSQ_l` and `results/TM_l` exist with a checkpoint directory inside and a config.

If you want to evaluate your own trained checkpoint, you dont have to adjust anything.

Now, to evaluate just run:
```bash
bash eval.sh
```

This should place the results in `results/test_top_p_05/vq_context_0`. If you want to modify CLIP models or context stroke lengths, you can do so in `generate_samples_stage2.py` and `eval_stage2_generations.py`.

## IconShop
If you want to eval IconShop, you first have to download the models. Go to the [IconShop GitHub page](https://github.com/kingnobro/IconShop#sample) and place the files in the `iconshop_checkpoints` directory that you may have to create first. The paths `iconshop_checkpoints/word_embedding_512.pt` and `iconshop_checkpoints/epoch_100/pytorch_model.bin` must exist.

Then, you can just do:
```bash
bash eval_iconshop.sh
```

The results are placed in `results/IconShop/output`.

## Im2Vec
From [the same Google drive](https://drive.google.com/drive/folders/1kEHkt0RVzHw7nwYADqAktYHlsVlWoFi7?usp=sharing), please download and unzip the Im2Vec.zip file so that the path `results/Im2Vec/figr8_full/config.yaml` exists.

Then run:
```bash
python eval_im2vec.py
```

# Sample / Inference
The easiest way to test these models interactively is by using the `inference.ipynb`. Please follow all the model downloading steps from the Evaluation section before trying this out. If all you want is seeing samples, then you could also just execute the eval scripts as they generate a bunch of SVGs.