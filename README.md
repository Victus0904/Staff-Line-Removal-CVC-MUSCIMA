# Staff-Line Removal on Handwritten Music (CVC-MUSCIMA)

Deep-learning staff-line removal for handwritten music scores, a key preprocessing step for
**Optical Music Recognition (OMR)**. Trained and evaluated on the CVC-MUSCIMA dataset.

## Results

| Metric | Score |
|---|---|
| IoU | **0.9910** |
| Dice coefficient | **0.9935** |
| Precision | 0.991 |
| Recall | 0.996 |

## Technical highlights

- **Architecture:** UNet for binary segmentation (staff vs. symbol pixels)
- **Loss:** weighted **Focal + Dice** loss, where Focal handles heavy class imbalance and Dice optimizes overlap
- **Training:** PyTorch Automatic Mixed Precision (AMP) for faster training and lower VRAM use
- **Augmentation:** Albumentations pipeline
- **Post-processing:** adaptive ink extraction and vertical erosion, which keep thin musical symbols intact and give clean, high-contrast output

## Run it

```bash
pip install -r requirements.txt
jupyter notebook staff-removal.ipynb
```

The dataset is downloaded automatically using `omrdatasettools`.
