import time
from pathlib import Path
import sys
import os
import cv2
project_root = Path("C:\Soham Projects\Machine Learning - Projects\PCB_Defect_Anaomaly")
from knn import knn_scoring
from Data.dataset import MVTecDataset
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import torch
from extractor import WideResNetFeatureExtractor   

data_path = project_root / "Data" / "mvtec"
embeddings_path = project_root / "memory_bank"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = WideResNetFeatureExtractor().to(device)

# n_split = 0.075
# memory_bank = torch.load(embeddings_path / f"memory_bank_bottle.pt", map_location="cpu")
# num_samples = int(n_split * len(memory_bank))
# indices = torch.randperm(len(memory_bank))[:num_samples]
# memory_bank_small = memory_bank[indices]

# pca_dim = 128
# pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=42)
# memory_bank_pca = pca.fit_transform(memory_bank_small.numpy())

# nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(memory_bank_pca)
# dataset = MVTecDataset(data_path / "bottle", split="test", return_mask=True)
# sample = dataset[1]

# extractor.eval()
# start = time.time()
# scores, heatmap = knn_scoring(sample["image"],memory_bank_pca, nn_model, extractor, pca)
# end = time.time()
# print(f"Time taken for KNN scoring: {end - start:.4f} seconds")

# sample = dataset[5]
# start = time.time()
# scores, heatmap = knn_scoring(sample["image"],memory_bank_pca, nn_model, extractor, pca)
# end = time.time()
# print(f"Time taken for KNN scoring: {end - start:.4f} seconds")

# evaluate_category("bottle", dataset_name="mvtec", extractor=extractor, n_split=0.075, pca_dim=128, batch_size=8)

# evaluate_dataset(dataset_name="mvtec", extractor=extractor, n_split=0.075, pca_dim=128, batch_size=8)