from pathlib import Path
import sys
import os
project_root = Path("C:\\Soham Projects\\Machine Learning - Projects\\PCB_Defect_Anaomaly")

from Data.dataset import MVTecDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
# data_path = project_root / "Data" / "mvtec"
output_path = project_root / "memory_bank"
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def create_memory_bank(Dataset, Feature_Extractor, data_path, batch_size=8):
    Extractor = Feature_Extractor()
    Extractor.to(device)
    for sample in os.listdir(data_path):
        sample_path = data_path / sample
        dataloader = DataLoader(Dataset(sample_path, split="train"), batch_size=batch_size)
        result_path = f"{output_path}/memory_bank_{sample}.pt"
        all_embeddings = []
        if os.path.exists(result_path):
            print(f"Memory bank for {sample} already exists. Skipping...")
            continue
        Extractor.eval()
        with torch.no_grad():
            for img_batch in tqdm.tqdm(dataloader):
                embeddings = Extractor(img_batch["image"].to(device))
                all_embeddings.append(embeddings.cpu())
            
            memory_bank = torch.cat(all_embeddings, dim=0)
            memory_bank = memory_bank.reshape(-1, memory_bank.shape[-1])
            torch.save(memory_bank.cpu(), result_path)