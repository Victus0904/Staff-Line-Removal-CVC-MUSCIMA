# Staff-Line Removal on Handwritten Music (CVC-MUSCIMA)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![Task](https://img.shields.io/badge/task-semantic%20segmentation-purple)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Deep-learning **staff-line removal** for handwritten music scores, a key preprocessing
step for **Optical Music Recognition (OMR)**. A UNet learns to separate musical symbols
from the five-line staff, so later stages can recognise notes, clefs and rests on a
clean image. It is trained and evaluated on the
[CVC-MUSCIMA](http://www.cvc.uab.es/cvcmuscima/index_database.html) dataset.

## Results

| Metric | Score |
|---|---|
| **IoU** | **0.9910** |
| **Dice coefficient** | **0.9935** |
| Precision | 0.991 |
| Recall | 0.996 |

## Approach

```mermaid
flowchart LR
    A[Score image] --> B[Albumentations<br/>augmentation]
    B --> C[UNet<br/>encoder–decoder]
    C --> D[Symbol mask]
    D --> E[Adaptive ink extraction<br/>+ vertical erosion]
    E --> F[Staff-free score]
```

- **Architecture:** UNet for binary segmentation (symbol pixels vs. staff and background)
- **Loss:** weighted **Focal + Dice**. Focal loss handles the heavy class imbalance (thin strokes on a mostly blank page); Dice optimises overlap directly.
- **Training:** PyTorch automatic mixed precision (AMP) for faster training and lower VRAM use, for 20 epochs
- **Augmentation:** an Albumentations pipeline that simulates scanning and handwriting variation
- **Post-processing:** adaptive ink extraction and vertical erosion keep thin musical symbols intact and give clean, high-contrast output

## Run it

```bash
git clone https://github.com/Victus0904/Staff-Line-Removal-CVC-MUSCIMA.git
cd Staff-Line-Removal-CVC-MUSCIMA
pip install -r requirements.txt
jupyter notebook staff-removal.ipynb
```

The dataset is downloaded automatically with
[`omrdatasettools`](https://github.com/apacha/OMR-Datasets). A CUDA GPU is recommended;
the notebook also runs on Kaggle and Colab.

## Contents

| File | Purpose |
|---|---|
| `staff-removal.ipynb` | Data download, dataset/augmentations, UNet, training loop, metrics, visualisations |
| `requirements.txt` | Dependencies |

## Acknowledgements

CVC-MUSCIMA: A. Fornés, A. Dutta, A. Gordo, J. Lladós, *"CVC-MUSCIMA: A Ground-Truth of
Handwritten Music Score Images for Writer Identification and Staff Removal"*, IJDAR 2012.

## License

Code: [MIT](LICENSE). The dataset is subject to its own terms.
