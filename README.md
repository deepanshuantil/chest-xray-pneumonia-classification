# Chest X-ray Pneumonia Classification with Deep Learning

Classifying chest X-ray images into **Normal**, **Bacterial Pneumonia** and **Viral Pneumonia** using transfer learning. Four pretrained architectures — **EfficientNet-B5**, **Vision Transformer (ViT-B/16)**, **ResNet-50** and **Xception** — were fine-tuned and compared on the same balanced dataset.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-torchvision%20%7C%20timm-red)
![Platform](https://img.shields.io/badge/Run%20on-Kaggle%20GPU%20%7C%20Colab-20BEFF)

---

> **🔧 Update — v2 leak-free pipeline:** while reviewing v1 I found that augmented copies were made *before* the train/test split, so the test set could contain altered copies of training images. [`chest_xray_classification_v2.ipynb`](notebooks/chest_xray_classification_v2.ipynb) fixes this (patient-level split first, augment training data only, clean eval transforms). The results below are from **v1**; v2 results will replace them once retrained on a GPU. See [What v2 fixes](#what-v2-fixes).

## Results (v1)

| Model | Framework | Test accuracy | Epochs (early stopping) |
|---|---|---|---|
| **Vision Transformer (ViT-B/16)** | PyTorch / torchvision | **87.65%** | 12 |
| EfficientNet-B5 | TensorFlow / Keras | 87.08% | 12 |
| Xception | PyTorch / timm | 87.05% | 14 |
| ResNet-50 | PyTorch / torchvision | 85.13% | 16 |

**Per-class results (Xception, test set of 834 images):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Normal | 0.96 | 0.97 | 0.97 |
| Bacterial Pneumonia | 0.86 | 0.84 | 0.85 |
| Viral Pneumonia | 0.83 | 0.84 | 0.83 |

**Key finding:** separating *Normal* from *Pneumonia* is easy for every model (~97% F1). The hard part is telling **bacterial and viral pneumonia apart** — almost all errors sit between those two classes, which matches how difficult that distinction is for radiologists from an X-ray alone.

<p align="center">
  <img src="images/xception_confusion_matrix.png" width="480" alt="Xception confusion matrix">
</p>

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,856 paediatric chest X-ray JPEGs (Kermany et al., 2018), originally labelled **Normal** vs **Pneumonia**.

### From 2 classes to 3
The original *Pneumonia* folder was split into **bacterial** and **viral** using the labels embedded in each file name (`..._bacteria_...`, `..._virus_...`), turning the task into a 3-class problem:

| Class | Images |
|---|---|
| Bacterial Pneumonia | 2,780 |
| Normal | 1,583 |
| Viral Pneumonia | 1,493 |

### Handling class imbalance
The minority classes (Normal, Viral) were oversampled with **Albumentations** (horizontal flip, ±20° rotation, brightness/contrast, Gaussian blur, random resized crop) until every class had **2,780 images** — 8,340 in total.

| Before | After |
|---|---|
| ![before](images/class_distribution_before.png) | ![after](images/class_distribution_after.png) |

The balanced set was split **70 / 15 / 15** into train / validation / test for the Keras model, and **80 / 10 / 10** for the PyTorch models.

---

## Approach

```
Raw X-rays ─► split Pneumonia into Bacterial / Viral ─► augment minority classes to 2,780 each
          ─► train / val / test split ─► fine-tune pretrained CNN / ViT ─► evaluate & compare
```

| | EfficientNet-B5 | ViT-B/16 | ResNet-50 | Xception |
|---|---|---|---|---|
| Pretrained on | ImageNet | ImageNet | ImageNet | ImageNet |
| Head | GAP → Dense(512, ReLU) → Dropout(0.3) → Dense(3) | Linear(768→3) | Linear(2048→3) | Linear(2048→3) |
| Optimiser | Adam, lr 1e-4 | AdamW, lr 5e-5, wd 5e-3 | AdamW, lr 1e-5, wd 1e-3 | AdamW, lr 5e-5, wd 5e-3 |
| LR schedule | ReduceLROnPlateau | Cosine annealing | ReduceLROnPlateau | Cosine annealing |
| Early stopping | val loss, patience 5 | val acc, patience 3 | val acc, patience 3 | val acc, patience 3 |

All models use 224×224 inputs, batch size 32, and on-the-fly augmentation during training (rotation, shifts, zoom, flips / colour jitter).

### Training curves

| EfficientNet-B5 | ViT-B/16 | ResNet-50 |
|---|---|---|
| ![](images/efficientnet_training_curves.png) | ![](images/vit_training_curves.png) | ![](images/resnet50_training_curves.png) |

### Sample images

| Normal | Viral pneumonia | Bacterial pneumonia |
|---|---|---|
| ![](images/samples_normal.png) | ![](images/samples_viral.png) | ![](images/samples_bacterial.png) |

---

## Repository structure

```
.
├── notebooks/
│   ├── chest_xray_classification.ipynb      # v1: original pipeline and results
│   └── chest_xray_classification_v2.ipynb   # v2: leak-free split, same 4 models, shared evaluation
├── images/                               # figures exported from the notebook
├── requirements.txt
└── README.md
```

## How to run

1. Download the dataset from Kaggle and place the `NORMAL` and `PNEUMONIA` folders under one input folder.
2. Open `notebooks/chest_xray_classification_v2.ipynb` on **Kaggle** with a GPU and Internet enabled. Set `INPUT_DIR` in the first code cell if your dataset path differs from `/kaggle/input/chest-xray-pneumonia/chest_xray`.
3. Install dependencies if needed: `pip install -r requirements.txt`
4. Run all cells top to bottom.

## What v2 fixes

| Problem in v1 | Fix in v2 |
|---|---|
| Augmented copies were created **before** the train/val/test split, so near-copies of one X-ray could be in both training and test | Split the **original** images first; augment the **training set only** — val/test hold untouched originals |
| Several X-rays come from the **same patient** | **Patient-level split** with `StratifiedGroupKFold`, checked by an assertion so no patient appears in two sets |
| PyTorch `random_split` gave val/test images the **training augmentations** | Separate folders and transforms: random augmentation for train, plain resize + normalise for val/test |
| Keras EfficientNet got 0–1 inputs, but it rescales internally and expects 0–255 | Raw 0–255 pixels fed to EfficientNet |
| Each model evaluated slightly differently | One shared evaluation for all models (accuracy, macro F1, per-class F1, confusion matrix) saved to `results/` |

v2 scores are expected to be somewhat **lower** than v1 — they are the honest numbers for how the model handles X-rays from new patients.

## Next steps

- Retrain v2 on Kaggle GPU and publish its results here.
- Bacterial vs viral confusion is the main error source — class-weighted loss, higher input resolution or Grad-CAM inspection could help.
- Add AUC/ROC, Grad-CAM explainability, and a simple Streamlit/Gradio demo.

> ⚠️ This is an educational project and is **not** a medical diagnostic tool.

## Tech stack

Python · TensorFlow / Keras · PyTorch · torchvision · timm · Albumentations · OpenCV · scikit-learn · NumPy · pandas · Matplotlib · Seaborn · Kaggle GPU · Google Colab

## Author

**Deepanshu** — feel free to connect on LinkedIn or open an issue with questions.
