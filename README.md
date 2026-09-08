# MDNAL-SingleCell-Spatial-Gene-Prediction
MDNAL for morphology-guided single-cell spatial gene-expression prediction from H&amp;E histology, integrating ConvNeXt-Tiny, neighborhood-expression auxiliary learning, and Pearson correlation-guided optimization on 10x Genomics Xenium breast cancer data.

# MDNAL: Morphology-Guided Deep Neighborhood-Aware Learning

Official PyTorch implementation of **Morphology-Guided Deep Neighborhood-Aware Learning (MDNAL)** for single-cell spatial gene-expression prediction from H&E histopathology.

MDNAL predicts spatial gene-expression profiles directly from cell-centered H&E image patches while using local neighborhood-expression information as auxiliary supervision during training. Importantly, spatial transcriptomic measurements are not required as model inputs during inference.

---

## Overview

Spatial transcriptomics enables high-resolution characterization of molecular tissue organization but remains considerably more resource-intensive than routine H&E histopathology. MDNAL investigates whether morphological information contained in H&E images can be used to predict single-cell spatial gene-expression patterns.

The framework combines:

- ConvNeXt-Tiny morphological feature extraction
- Morphology-guided gene selection
- Neighborhood-expression auxiliary learning
- Pearson correlation-guided optimization
- H&E-only inference
- Independent cross-section evaluation

---

## Framework

For each cell, a **512 × 512 H&E image patch** is extracted around its spatial coordinate and resized to **224 × 224 pixels**.

The image is processed using an ImageNet-pretrained **ConvNeXt-Tiny** backbone followed by a shared feature-projection network.

The learned representation is used by three branches:

1. **Gene-Expression Prediction Head**  
   Predicts the expression levels of the selected morphology-associated genes.

2. **Neighborhood-Expression Auxiliary Head**  
   Predicts the average expression profile of the eight nearest spatial neighbors during training.

3. **Morphology Embedding Head**  
   Provides additional regularization of the learned morphological representation.

Neighborhood-expression measurements are used **only as auxiliary supervision during training** and are not supplied as model inputs. Therefore, the trained MDNAL framework requires only an H&E image patch during inference.

---

## Dataset

Experiments use the **10× Genomics Xenium FFPE Human Breast Cancer** dataset publicly available through the NCBI Gene Expression Omnibus (GEO).

- **GSE243168** — Xenium breast cancer study
- **GSM7780153 (Rep1)** — model development
- **GSM7780154 (Rep2)** — independent cross-section testing

Rep1 is used for model development, whereas Rep2 remains completely independent for evaluating cross-section generalization.

### Dataset Links

**GSE243168**

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243168

**Rep1 — GSM7780153**

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7780153

**Rep2 — GSM7780154**

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7780154

---

## Data Preprocessing

The preprocessing pipeline includes:

- Cell-coordinate alignment with H&E histology
- Cell-centered H&E patch extraction
- Patch-quality filtering
- Gene-expression normalization
- ImageNet image normalization
- Training-time image augmentation
- Spatial nearest-neighbor construction

Each cell-centered patch is initially extracted at:

```text
512 × 512 pixels
```

and subsequently resized to:

```text
224 × 224 pixels
```

for network input.

Local spatial neighborhoods are constructed using:

```text
k = 8
```

nearest cells based on spatial coordinates.

Training-time image augmentation includes:

- Random horizontal flipping
- Random vertical flipping
- 90° rotational augmentation

---

## MDNAL Architecture

The implementation uses an ImageNet-pretrained **ConvNeXt-Tiny** backbone for morphological feature extraction.

Most backbone parameters are frozen, while the final ConvNeXt stage is fine-tuned.

The shared feature representation follows:

```text
H&E Patch
   ↓
ConvNeXt-Tiny
   ↓
Global Average Pooling
   ↓
Linear → 512
   ↓
Batch Normalization
   ↓
GELU
   ↓
Dropout (0.20)
   ↓
Linear → 256
   ↓
Batch Normalization
   ↓
GELU
   ↓
Dropout (0.10)
   ↓
Shared Morphological Representation
```

The shared representation feeds three task-specific branches:

```text
                         ┌───────────────────────────────┐
                         │ Gene-Expression Head          │
                         │ 256 → 100 Genes               │
                         └───────────────────────────────┘
                                      ↑
                                      │
H&E → ConvNeXt-Tiny → Shared MLP ─────┼────────────────────
                                      │
                         ┌───────────────────────────────┐
                         │ Neighborhood Auxiliary Head   │
                         │ 256 → 100 Genes               │
                         └───────────────────────────────┘
                                      │
                         ┌───────────────────────────────┐
                         │ Morphology Embedding Head     │
                         │ 256 → 64 → 16                 │
                         └───────────────────────────────┘
```

The neighborhood-expression branch provides **training-only auxiliary supervision**.

At inference:

```text
H&E Patch → Trained MDNAL → 100-Gene Expression Prediction
```

No neighborhood-expression or Xenium measurement is required as model input during inference.

---

## Morphology-Guided Gene Selection

Gene-expression predictability from H&E histology varies substantially across genes because not all transcriptional signals are equally associated with observable tissue morphology.

MDNAL therefore focuses prediction on a selected set of:

```text
100 morphology-associated genes
```

Gene selection is performed using the model-development data, while the independent Rep2 tissue section remains excluded from model development.

---

## Neighborhood-Aware Learning

For each cell, MDNAL constructs a local neighborhood using the:

```text
k = 8
```

nearest cells in spatial coordinate space.

The normalized expression profiles of neighboring cells are averaged to generate a neighborhood-expression target.

This information is used exclusively as **auxiliary supervision during training**.

The neighborhood-expression vector is:

- NOT concatenated with H&E features
- NOT supplied as a model input
- NOT required during inference

This allows the model to learn morphology associated with local tissue context while preserving **H&E-only prediction**.

---

## Training Configuration

The model is optimized using **AdamW** under a partial backbone fine-tuning strategy.

| Parameter | Configuration |
|---|---|
| Backbone | ConvNeXt-Tiny |
| Initialization | ImageNet pretrained |
| Fine-tuning | Final ConvNeXt stage |
| Optimizer | AdamW |
| Initial learning rate | 5 × 10⁻⁵ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 32 |
| Epochs | 15 |
| LR scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 5 epochs |
| Minimum learning rate | 1 × 10⁻⁶ |
| Mixed precision | AMP |
| Gradient clipping | 1.0 |
| Patch size | 512 × 512 |
| Network input | 224 × 224 |
| Neighborhood size | k = 8 |
| Prediction targets | 100 genes |

---

## Optimization Objective

MDNAL jointly optimizes gene-expression prediction and neighborhood-expression auxiliary learning.

The primary gene-expression objective combines:

- Smooth L1 regression loss
- Pearson correlation-guided loss

The neighborhood-expression auxiliary objective also combines regression and correlation-based optimization.

The overall objective encourages the network to preserve both:

- Single-cell morphology–expression correspondence
- Local neighborhood-associated transcriptional information

---

## Experimental Protocol

The experiments follow a strict cross-section evaluation strategy.

### Model Development

```text
Xenium Rep1 — GSM7780153
```

Following preprocessing and quality filtering, Rep1 is divided into:

```text
Training   = 80%
Validation = 20%
Random seed = 42
```

The checkpoint achieving the highest validation **Mean PCC** is selected.

### Independent Testing

```text
Xenium Rep2 — GSM7780154
```

Rep2 remains independent from model development and is used exclusively for cross-section testing.

This experimental design prevents information leakage between model development and independent evaluation.

---

## Evaluation Metrics

Prediction performance is evaluated using:

- Mean Pearson Correlation Coefficient (Mean PCC)
- Median Pearson Correlation Coefficient (Median PCC)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Gene-wise Pearson correlation
- Gene-wise Spearman correlation
- Gene-wise R²

Additional analyses include:

- Gene-level prediction analysis
- Predicted-versus-measured expression plots
- Spatial expression reconstruction
- Biological gene-category analysis
- Grad-CAM visualization
- Ablation analysis
- Computational efficiency analysis

---

## Main Results

Under the Rep1 → Rep2 independent cross-section evaluation protocol, MDNAL achieves:

| Metric | Result |
|---|---:|
| Mean PCC | **0.5475** |
| Median PCC | **0.5234** |
| RMSE | **1.9097** |
| MAE | **1.3030** |
| Number of target genes | **100** |

Several epithelial and tumor-associated genes demonstrate strong morphology-associated prediction performance, including:

- **KRT7**
- **FASN**
- **FOXA1**
- **EPCAM**
- **KRT8**
- **ERBB2**

The highest observed gene-wise Pearson correlation reaches approximately:

```text
PCC = 0.7988
```

---

## Notebook

The complete experimental workflow is provided in:

```text
notebooks/MDNAL_Xenium.ipynb
```

The notebook covers:

1. Dataset loading
2. Xenium and H&E alignment
3. Gene-expression preprocessing
4. Cell-centered patch extraction
5. Patch-quality filtering
6. Morphology-guided target preparation
7. Spatial neighborhood construction
8. Dataset and DataLoader preparation
9. ConvNeXt-Tiny model construction
10. MDNAL prediction heads
11. Multi-objective loss computation
12. Model training
13. Validation and checkpoint selection
14. Independent Rep2 evaluation
15. Gene-level correlation analysis
16. Spatial expression visualization
17. Grad-CAM analysis
18. Result table generation
19. Figure generation

---

## Requirements

The implementation uses Python and PyTorch.

Core dependencies include:

```text
torch
torchvision
timm
numpy
pandas
scipy
scikit-learn
scanpy
opencv-python
tifffile
matplotlib
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```text
MDNAL-SingleCell-Spatial-Gene-Prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   └── MDNAL_Xenium.ipynb
│
├── figures/
│
└── results/
```

---

## Reproducibility

The experimental protocol maintains strict separation between model development and independent evaluation.

```text
Rep1
 ├── Training: 80%
 └── Validation: 20%

Rep2
 └── Independent Testing
```

Key reproducibility settings:

```text
Random seed: 42
Epochs: 15
Batch size: 32
Neighborhood size: 8
Input size: 224 × 224
Target genes: 100
Model selection: Highest validation Mean PCC
```

---

## Citation

If you use this repository or MDNAL framework in your research, please cite the corresponding paper:

```bibtex
@article{mdnal2026,
  title   = {Morphology-Guided Single-Cell Spatial Gene Expression Prediction from Histology Using Deep Neighborhood-Aware Learning},
  author  = {Authors},
  journal = {Journal},
  year    = {2026}
}
```

Citation information will be updated after publication.

---

## License

Please refer to the repository `LICENSE` file for usage and distribution terms.

---

## Acknowledgements

This implementation uses **PyTorch**, **timm**, **Scanpy**, and the publicly available **10× Genomics Xenium FFPE Human Breast Cancer** spatial transcriptomics dataset.
