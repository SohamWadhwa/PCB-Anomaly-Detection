# PCB Anomaly Detection

An anomaly-detection project for industrial visual inspection using a PatchCore-style pipeline on the **MVTec AD** dataset.

## What this project does

- Extracts patch embeddings with a pretrained **WideResNet-50-2** backbone.
- Builds a **memory bank** from normal training images.
- Scores test images with **PCA + 1-NN (KNN)** distance in embedding space.
- Produces:
  - image-level anomaly score/prediction
  - pixel-level anomaly map (heatmap)
- Includes a desktop UI (`PatchSim.py`) for interactive inference.

## Repository structure

- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/Data/dataset.py` – MVTec dataset loader (train/test, optional masks)
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/Data/transformations.py` – image and mask transforms
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/extractor.py` – WideResNet feature extractor
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/memory_bank.py` – memory-bank creation function
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/knn.py` – KNN scoring and evaluation helpers
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/PatchSim.py` – PySide6 GUI app
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/notebooks/` – notebook-based walkthroughs
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/results/` – saved evaluation summaries

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Current `requirements.txt` includes:

- scikit-learn
- pandas
- numpy
- matplotlib
- torch / torchvision (CUDA 12.1 index)

> The GUI also requires **PySide6**, **opencv-python**, **Pillow**, and **tqdm**.

## Data setup

The repository ignores `Data/mvtec/` by default, so download and place the MVTec AD dataset like:

```text
Data/
  mvtec/
    bottle/
    cable/
    ...
```

## Important path configuration

Several scripts use a hardcoded Windows path via `project_root`.

Update `project_root` in these files before running:

- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/PatchSim.py`
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/knn.py`
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/memory_bank.py`
- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/test.py`

Set it to your local repository path.

## Typical workflow

1. **Build memory bank** from normal training images using functions in `memory_bank.py` (or notebook `03_memory_bank.ipynb`).
2. **Evaluate** categories with helpers in `knn.py` (or notebook `04_knn_scoring.ipynb`).
3. **Run GUI inference**:

```bash
python PatchSim.py
```

Then select dataset/category and upload an image.

## Notebooks

- `01_dataset.ipynb` – dataset loading and inspection
- `02_extractor.ipynb` – feature extraction exploration
- `03_memory_bank.ipynb` – memory-bank generation
- `04_knn_scoring.ipynb` – anomaly scoring and AUROC evaluation

## Available results

Example summary is available in:

- `/home/runner/work/PCB-Anomaly-Detection/PCB-Anomaly-Detection/results/knn_results_summary_5.csv`

This file contains per-category metrics such as:

- feature extraction time
- KNN search time
- image AUROC
- pixel AUROC
- PCA dimension and split ratio

## Notes

- `model.py` is currently empty.
- This repository appears to be a research/prototype codebase focused on MVTec AD-style anomaly detection.
