# CutMix vs Cutout vs Mixup — Data Augmentation Benchmark on CIFAR-10

A clean, reproducible benchmark comparing three popular data augmentation strategies against a baseline using **ResNet-18** on **CIFAR-10**, trained for 30 epochs under identical conditions.

---

## Results

| Method   | Best Test Accuracy | Final Train Accuracy | Technique                  |
|----------|--------------------|----------------------|----------------------------|
| Baseline | 93.00%             | 99.13%               | Random crop + HFlip only   |
| Cutout   | 93.19%             | 93.99%               | Zero-masking square patch  |
| **Mixup**    | **93.65%**         | 54.26%               | Linear image interpolation |
| CutMix   | 92.99%             | 74.26%               | Paste crop + label mixing  |

**Winner: Mixup** with **93.65%** test accuracy on CIFAR-10.

---

## What Are These Augmentations?

| Method | Core Idea | Paper |
|--------|-----------|-------|
| **Mixup** | Linearly interpolates two images and their labels using a Beta-distributed coefficient λ | Zhang et al., 2018 |
| **Cutout** | Masks a random square region of the input image with zeros during training | DeVries & Taylor, 2017 |
| **CutMix** | Pastes a cropped patch from one image onto another; labels are mixed proportionally to patch area | Yun et al., 2019 |

---

## Augmentation Visual Comparison

Original, Mixup (λ=0.5), Cutout (16x16), and CutMix (λ=0.5) applied to the same CIFAR-10 image.

<img width="1339" height="362" alt="image" src="https://github.com/user-attachments/assets/3d86bc9b-eb36-404d-a8c6-53c81a3fe895" />

---

## Training Curves

Validation loss and validation accuracy across all 30 epochs for each method.

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/76da71ff-64e8-47f4-8dc7-8dea283106b4" />


**Observations:**
- Baseline and Cutout converge quickly and cleanly with standard loss behaviour.
- Mixup shows a characteristically high training loss (because it never sees clean labels) but steadily improves validation accuracy, eventually surpassing all other methods.
- CutMix converges more slowly in the early epochs but catches up by epoch 25-30, suggesting it may benefit from longer training runs.

---

## Test Accuracy Comparison

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/c8e24cbd-5c58-400d-aa0e-ad9ba29b23b1" />


---

## Key Findings

- **Mixup** achieved the highest test accuracy (93.65%) by forcing the model to behave linearly between training examples — reducing overconfidence and improving generalisation.
- **Cutout** is the most practical option: no label manipulation, trains at baseline speed, and still improves over the baseline by forcing context-aware feature learning.
- **CutMix** converges more slowly early on but catches up — it may benefit more from longer training (>30 epochs).
- **Baseline overfits significantly** — training accuracy (99.13%) far exceeds validation accuracy (93.00%), confirming the value of augmentation.
- **Label-mixing methods (Mixup/CutMix)** show lower training accuracy intentionally — they never see clean labels, so training accuracy is not a fair metric for these methods.

---

## Setup and Reproducibility

### Requirements

```bash
pip install torch torchvision numpy matplotlib
```

Tested on: Python 3.10, PyTorch 2.x, CUDA (Tesla T4 on Google Colab)

### Run

Open the notebook in Google Colab (recommended for GPU access):

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16l5pGx-F29CSHW5wqhKsmHnXHFQV6jDK)

Or clone and run locally:

```bash
git clone https://github.com/Vedantt207/resnet18-augmentation-study.git
cd resnet18-augmentation-study
jupyter notebook cutmix_benchmark.ipynb

```

---

## Model Architecture

ResNet-18 adapted for CIFAR-10's small 32x32 images:
- `conv1` replaced: 7x7 kernel to 3x3 (stride 1, padding 1)
- `maxpool` removed (replaced with `nn.Identity`)
- Final FC layer outputs 10 classes
- Total parameters: **11,173,962**

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Epochs | 30 |
| Learning Rate | 0.1 (Cosine Annealing) |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4) |
| Mixup / CutMix alpha | 1.0 (Beta distribution) |
| Cutout length | 16x16 pixels |
| Random seed | 42 |

---

## Project Structure

```
cutmix-benchmark/
│
├── cutmix_benchmark.ipynb      # Main notebook with all experiments
└── README.md
```

---

## References

- Zhang et al. (2018) — [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- DeVries & Taylor (2017) — [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)
- Yun et al. (2019) — [CutMix: Training Strategy that Makes Use of Sample Pastes](https://arxiv.org/abs/1905.04899)
- He et al. (2016) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

## License

MIT License — free to use, modify, and distribute.
