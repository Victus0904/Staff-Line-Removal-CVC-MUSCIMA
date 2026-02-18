# Staff-Line-Removal-on-Handwritten-Music-CVC-MUSCIMA-

This project implements a Deep Learning solution for removing staff lines from handwritten music scores using the CVC-MUSCIMA dataset. The model achieves high-precision segmentation to facilitate Optical Music Recognition (OMR) tasks.

🚀 Key Performance
IoU: 0.9910

Dice Coefficient: 0.9935

Precision: 0.991

Recall: 0.996

🛠️ Technical Highlights
Architecture: UNet-based binary segmentation.

Loss Function: A weighted combination of Focal Loss (to handle class imbalance) and Dice Loss (to optimize overlap).

AMP Training: Utilized PyTorch Automatic Mixed Precision (AMP) for faster training and reduced VRAM usage.

Post-Processing: Implemented Adaptive Ink Extraction and Vertical Erosion to preserve thin musical symbols while ensuring dark, high-contrast outputs.

📊 Results
💻 Installation
Bash
```
pip install -r requirements.txt
```
