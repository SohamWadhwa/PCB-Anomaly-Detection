import time
from pathlib import Path
import sys
import os
project_root = Path("C:\Soham Projects\Machine Learning - Projects\PCB_Defect_Anaomaly")

from Data.dataset import MVTecDataset
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import tqdm

data_path = project_root / "Data" / "mvtec"
embeddings_path = project_root / "memory_bank"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def knn_scoring(image, memory_bank, knn_model, extractor, pca):
    image = image.unsqueeze(0).to(device)
    start = time.time()
    with torch.no_grad():
        embeddings = extractor(image).cpu().numpy()
    end = time.time()
    print(f"Feature extraction time: {end - start:.4f} seconds")
    B, P, D = embeddings.shape
    patch = embeddings.reshape(-1, D)
    patch_pca = pca.transform(patch)  
    start = time.time()
    distances, _ = knn_model.kneighbors(patch_pca)
    end = time.time()
    print(f"KNN search time: {end - start:.4f} seconds")
    scores = distances.squeeze()
    anomaly_map = scores.reshape(32, 32)
    heatmap = cv2.resize(anomaly_map, (256, 256))
    return scores, heatmap

def evaluate_category(category, dataset_name="mvtec", extractor=None, n_split=0.1, pca_dim=64, batch_size=8):
    if dataset_name == "mvtec":
        dataset = MVTecDataset(data_path / category, split="test", return_mask=True)
    
    memory_bank = torch.load(embeddings_path / f"memory_bank_{category}.pt", map_location="cpu")
    num_samples = int(n_split * len(memory_bank))
    indices = torch.randperm(len(memory_bank))[:num_samples]
    memory_bank_small = memory_bank[indices]

    pixel_gt = []
    pixel_pred = []

    image_gt = []
    image_pred = []

    extractor_time = []
    knn_time = []

    dataset = MVTecDataset(data_path / category, split="test", return_mask=True)

    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=42)
    memory_bank_pca = pca.fit_transform(memory_bank_small.numpy())

    nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(memory_bank_pca)

    extractor.to(device)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    extractor.eval()
    with torch.no_grad():
        for sample in tqdm.tqdm(test_loader):
            images = sample["image"].to(device, non_blocking=True)
            masks = sample["mask"].cpu()
            labels = sample["label"].cpu()

            start = time.time()
            embeddings = extractor(images)
            end = time.time()
            extractor_time.append(end - start)

            B, P, D = embeddings.shape
            all_patches = embeddings.cpu().numpy().reshape(-1, D)
            all_patches_pca = pca.transform(all_patches)

            start = time.time()
            distances, _ = nn_model.kneighbors(all_patches_pca)
            end = time.time()
            knn_time.append(end - start)
            distances = distances.reshape(B, P)

            for i in range(B):
                scores = distances[i]

                image_score = np.partition(scores, -50)[-50:].mean()
                anomaly_map = scores.reshape(32, 32)
                heatmap = cv2.resize(anomaly_map, (256, 256))

                pixel_gt.extend(masks[i].squeeze().numpy().flatten())
                pixel_pred.extend(heatmap.flatten())

                image_gt.append(labels[i].item())
                image_pred.append(image_score)

    print(f"Average feature extraction time per batch: {np.mean(extractor_time):.4f} seconds")
    print(f"Average KNN search time per batch: {np.mean(knn_time):.4f} seconds")   

    image_auc = roc_auc_score(image_gt, image_pred)
    pixel_auc = roc_auc_score(pixel_gt, pixel_pred)

    print("Image AUROC:", image_auc)
    print("Pixel AUROC:", pixel_auc)

    return { "category": category, "image_auroc": image_auc, "pixel_auroc": pixel_auc, "feature_extraction_time": np.mean(extractor_time), "knn_search_time": np.mean(knn_time), "n_split": n_split, "pca_dim": pca_dim }

def evaluate_dataset(dataset_name="mvtec", categories=None, extractor=None, n_split=0.1, pca_dim=64, batch_size=8):
    if dataset_name == "mvtec":
        if categories is None:
            categories = os.listdir(data_path)
    
    results = []
    for category in categories:
        print(f"Evaluating category: {category}")
        result = evaluate_category(category, dataset_name=dataset_name, extractor=extractor, n_split=n_split, pca_dim=pca_dim, batch_size=batch_size)
        results.append(result)
    
    return results