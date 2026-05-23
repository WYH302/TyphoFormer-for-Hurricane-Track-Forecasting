# 🏆[SIGSPATIAL 25 Best Short Paper Award] TyphoFormer
Language-Augmented Transformer for Accurate Typhoon (Hurricane) Track Forecasting

## 🫶 How to Cite:
> If you find our work useful, please kindly cite our paper, thank you for your appreciation!

```
@inproceedings{lityphoformer2025,
author = {Li, Lincan and Ozguven, Eren Erman and Zhao, Yue and Wang, Guang and Xie, Yiqun and Dong, Yushun},
title = {TyphoFormer: Language-Augmented Transformer for Accurate Typhoon Track Forecasting},
booktitle={33rd ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM SIGSPATIAL 2025)},
location = {Minnesota, MN, USA},
url = {https://doi.org/10.1145/3748636.3763223},
year = {2025}
}
```

## 🧭 1.Project Overview
> TyphoFormer is a hybrid multi-modal Transformer designed for tropical cyclone (other names: Hurricane, Typhoon) track prediction. It integrates `numerical meteorological features` and `LLM-augmented language embeddings` through a Prompt-aware Gating Fusion (PGF) module, followed by a spatio-temporal Transformer backbone and autoregressive decoding for track forecasting.


## 🧱 2.Repository Structure
```bash
TyphoFormer/
├── model/
│   ├── STTransformer.py       # Spatio-Temporal backbone
│   ├── PGF_module.py          # Prompt-aware Gating Fusion module
│   ├── TyphoFormer.py         # TyphoFormer model architecture
│
│
├── data/                      # Processed Typhoon datasets in '.npy' files
│   ├── train/                 # contains `train_part1.zip` and `train_part2.zip`. Unzip and put all `.npy` files under "train" folder directly.
│   ├── val/
│   └── test/                  # contains `test.zip`. Unzip to get all the `.npy` files.
│
├── embedding_chunks/          # LLM generated semantic descriptions are embedded by sentence-transformer
│   ├── emb_chunk_000.npy
│   ├── ......
│   ├── emb_chunk_006.npy ...
│
├── HURDAT_2new_3000.csv       # Raw typhoon dataset, includes 5 years' typhoon data here as an example
├── generate_text_description_new.py   # GPT-based language generation
├── generate_text_embeddings.py        # Embedding generation via MiniLM-L6-v2
├── prepare_typhoformer_data.py        # Dataset preparation script
├── train_typhoformer.py               # Training entry point
├── eval_typhoformer.py                # Evaluation script
└── utils.py
```

## ⚙️ 3. Environment Setup
```
torch >= 2.1.0
transformers
sentence-transformers
openai
tqdm
pandas
numpy
```

## 🧩 4. Data Preparation

(1) Step 1: Use `generate_text_description_new.py` to create GPT-4o enhanced natural language descriptions for each typhoon record. (We already provided the generated language descriptions with this repository).

(2) Step 2: Convert textual descriptions to embeddings using `generate_text_embeddings.py` (model: MiniLM).

(3) Step 3: Combine numerical and textual embeddings into ready-to-use dataset using `prepare_typhoformer_data.py`.

(4) Step 4: The final dataset is stored under:
```
data/train/xxx.npy
data/val/yyy.npy
data/test/zzz.npy
```
### ❗️[NOTICE]

- **In this repository, we already provide five-year ground-truth typhoon records from HURDAT2, and the corresponding GPT-4o generated language descriptions, as well as the MiniLM generated language embeddings for you to try. However, in our own experiments, we use over 20+ years' Typhoon records and LLM-generated natural language descriptions as our database.**

- The raw numerical typhoon records from 2020-2024 is provided in `HURDAT_2new_3000.csv`
- If you want to generate your own language context descriptions using GPTs, make sure you have a valid OpenAI API Key and put it in the `generate_text_description_new.py`.

Each `.npy`file contains one piece of typhoon track record formatted as:
```
data = np.load(path, allow_pickle=True).item()
X = data["input"]
Y = data["target"]
```

## 🚀 5.Training and Evaluation

> 😄 We already provide a 5-year processed dataset, which can be used directly for model training and evaluation. 

```bash
# Train
python train_typhoformer.py

# Evaluate
python eval_typhoformer.py

```
>Training logs will be saved automatically under /checkpoints.
> You can adjust model training-related configurations in `train_typhoformer.py`:
```bash
# <Adjustable Configurations>
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
SAVE_DIR = "checkpoints"

BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_LEN = 12
PRED_LEN = 1
D_NUM = 14
D_TEXT = 384 #dim of language embedding (all-MiniLM-L6-v2）
```

<p align="center">
  <img src="https://github.com/LabRAI/TyphoFormer/blob/main/assets/test_code_visualization.gif" alt="code demo" width="800">
</p>


## 📊 6.Performance Results
<img width="600" alt="image" src="https://github.com/LabRAI/TyphoFormer/blob/main/assets/Table1_Experimental_results.png">


<img width="500" alt="image" src="https://github.com/LabRAI/TyphoFormer/blob/main/assets/MILTON_Track_Prediction.png">

