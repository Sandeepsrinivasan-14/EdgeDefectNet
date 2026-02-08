# EdgeDefectNet – Wafer Edge Defect Classification

## Overview
EdgeDefectNet is a deep learning–based wafer edge defect classification system built on a modified **ResNet-18** architecture.  
The model classifies **grayscale wafer edge / wafer map images** into **9 defect categories**, enabling automated defect screening for semiconductor manufacturing workflows.

The project focuses on robustness, reproducibility, and deployability, including support for **ONNX export**.

---

## Defect Classes
The model predicts one of the following 9 classes:

- Clean  
- Shorts  
- Opens  
- Bridges  
- Vias  
- Scratches  
- Cracks  
- LER (Line Edge Roughness)  
- Other  

---

## Dataset (Phase-1)

### Dataset Composition
- Train: 4,384 images  
- Validation: 1,096 images  
- Test: 1,183 images  

During training, a fixed fraction of the training data is reserved internally as a validation subset for early stopping and best-checkpoint selection. The Test split is held out and used only for final evaluation.

### Folder Structure (Submission ZIP)

`edge_defect_phase1_dataset.zip`
```text
+-- Train/
¦   +-- Clean/
¦   +-- Shorts/
¦   +-- Opens/
¦   +-- Bridges/
¦   +-- Vias/
¦   +-- Scratches/
¦   +-- Cracks/
¦   +-- LER/
¦   +-- Other/
+-- Validation/
¦   +-- Clean/
¦   +-- Shorts/
¦   +-- Opens/
¦   +-- Bridges/
¦   +-- Vias/
¦   +-- Scratches/
¦   +-- Cracks/
¦   +-- LER/
¦   +-- Other/
+-- Test/
    +-- Clean/
    +-- Shorts/
    +-- Opens/
    +-- Bridges/
    +-- Vias/
    +-- Scratches/
    +-- Cracks/
    +-- LER/
    +-- Other/
Each <class_name> corresponds to one of the 9 defect classes listed above.

Image Properties
Image type: Grayscale wafer-edge or wafer-map images

Format: .png (and similar lossless formats)

Resolution: 224 × 224 (applied via preprocessing)

Environment Setup
Create and Activate Virtual Environment
bash
python -m venv venv
venv\Scripts\activate    # Windows
Install Dependencies
Using requirements.txt:

bash
pip install -r requirements.txt
Or install manually:

bash
pip install torch torchvision scikit-learn pillow onnx onnxscript
Note: onnx and onnxscript are only required if ONNX export is needed.

Model and Training (Phase-1)
Backbone: ResNet-18 (ImageNet pretrained, modified for 1-channel input)

Input size: 1 × 224 × 224 grayscale images

Output: 9-class linear classifier head

Loss function: Cross-Entropy

Optimizer: Adam

Random seed: 42

Training platform: Windows 11, Python 3.11

Hardware: CPU-only training and inference

Training Command
bash
python train_phase1.py
The training script uses the Train split and monitors performance on the Validation split (created from the training data) for early stopping and model selection.

Best checkpoint is saved at:

text
checkpoints/phase1_best_resnet18.pt
Inference (Single Image)
Run inference on a single image:

bash
python -m scripts.infer_phase1 --image path\to\image.png
Example:

bash
python -m scripts.infer_phase1 --image .\dataset\Test\Shorts\MULTI_CLASS__image_Center_12015_png_jpg.rf.88f63a2bceb47b58fb93b4d118ee3738.png
This performs:

Model loading

Training-consistent preprocessing

Per-class probability estimation

Final predicted class output

Export to ONNX
Export the trained PyTorch model to ONNX:

bash
python export_onnx.py
Export Details
Output file: edge_defect_phase1.onnx

Opset version: 18

Approximate size: ~0.08 MB

The script rebuilds the model using build_model from scripts.infer_phase1 and loads the trained checkpoint.

Results (Test Set)
All metrics below are computed exclusively on the held-out Test set (dataset/Test), which is never used during training or validation.

Evaluation performed on the held-out test set (dataset/Test).

Accuracy: 0.9814

Macro Precision: 0.9474

Macro Recall: 0.9824

Confusion Matrix
text
[[132   0   0   0   0   0   0   0   0]
 [  0 130   0   1   0   0   1   0   2]
 [  0   0 153   0   1   2   0   0   0]
 [  0   0   0 224   1   0   0   0   0]
 [  0   0   1   1 128   0   2   1   0]
 [  3   0   0   1   0 129   0   0   0]
 [  0   2   0   0   1   0 130   0   0]
 [  0   0   0   0   0   0   2 131   0]
 [  0   0   0   0   0   0   0   0   4]]
Classification Report
text
              precision    recall  f1-score   support
Clean           0.98      1.00      0.99       132
Shorts          0.98      0.97      0.98       134
Opens           0.99      0.98      0.99       156
Bridges         0.99      1.00      0.99       225
Vias            0.98      0.96      0.97       133
Scratches       0.98      0.97      0.98       133
Cracks          0.96      0.98      0.97       133
LER             0.99      0.98      0.99       133
Other           0.67      1.00      0.80         4

Accuracy                              0.98      1183
Macro Avg        0.95      0.98      0.96      1183
Weighted Avg     0.98      0.98      0.98      1183
Repository Structure
text
EdgeDefectNet/
+- scripts/
|  +- infer_phase1.py
|  +- config_phase1.py
|  +- dataloaders_phase1.py
+- checkpoints/
|  +- phase1_best_resnet18.pt
|  +- phase1_class_names.py
+- export_onnx.py
+- compute_phase1_metrics.py
+- dataset/                  # Folder structure tracked via .gitkeep, images ignored
+- edge_defect_phase1_dataset.zip
+- README.md
+- requirements.txt
Note: Large datasets are not tracked in Git. Use .gitignore for dataset images.

Reproducing Phase-1 Results
bash
python train_phase1.py
python compute_phase1_metrics.py
python export_onnx.py
python -m scripts.infer_phase1 --image path\to\test_image.png
License
MIT License – see LICENSE file for details.

Contact
Author: Sandeep Srinivasan
Email: sndpsrinivasan@gmail.com
GitHub: https://github.com/Sandeepsrinivasan-14/EdgeDefectNet