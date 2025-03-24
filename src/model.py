#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

output_dir = Path('/mnt/output/task2')
input_dir = Path('/mnt/output/task1')
NOISE_COEFFICIENT = 0.1
MARGIN = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TransactionDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data).to(device)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor = self.data[idx]
        positive = anchor + torch.randn_like(anchor, device=device) * NOISE_COEFFICIENT
        negative_idx = np.random.randint(0, len(self.data))
        negative = self.data[negative_idx]
        return anchor, positive, negative

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.mean(pos_dist + torch.clamp(self.margin - neg_dist, min=0))
        return loss

class AnomalyDetector:
    def __init__(self, input_dim, embedding_dim=32, learning_rate=0.001):
        self.device = device
        self.encoder = Encoder(input_dim, embedding_dim).to(self.device)
        self.criterion = ContrastiveLoss().to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.scaler = torch.cuda.amp.GradScaler()  # Using torch.cuda.amp for mixed precision
        self.centroid = None
    
    def train(self, data_loader, num_epochs=50):
        self.encoder.train()
        all_embeddings = []
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in data_loader:
                anchor, positive, negative = batch[:3]
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    anchor_emb = self.encoder(anchor)
                    pos_emb = self.encoder(positive)
                    neg_emb = self.encoder(negative)
                    loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                if epoch == num_epochs - 1:
                    all_embeddings.append(anchor_emb.detach())
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.centroid = torch.mean(all_embeddings, dim=0)
    
    def get_embeddings(self, data):
        self.encoder.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            with torch.cuda.amp.autocast():
                embeddings = self.encoder(data_tensor)
        return embeddings.cpu().numpy()
    
    def detect_anomalies(self, data, percentile=95):
        self.encoder.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            with torch.cuda.amp.autocast():
                embeddings = self.encoder(data_tensor)
            
            if self.centroid is None:
                raise ValueError("Model must be trained first to compute centroid")
            
            distances = torch.norm(embeddings - self.centroid, dim=1)
            scores = distances.cpu().numpy()
        
        threshold = np.percentile(scores, percentile)
        anomalies = scores > threshold
        return scores, anomalies

if __name__ == "__main__":
    # Load data from CSV
    df = pd.read_csv(input_dir / 'kyc_full.csv')
    customer_ids = df['customer_id'].values
    data = df.drop('customer_id', axis=1).values
    num_features = data.shape[1]

    # Normalize data
    scaler_obj = StandardScaler()
    data_scaled = scaler_obj.fit_transform(data)

    # Create dataset and dataloader
    dataset = TransactionDataset(data_scaled)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train model
    epochs = 200
    detector = AnomalyDetector(input_dim=num_features)
    detector.train(data_loader, num_epochs=epochs)

    # Get embeddings
    embeddings = detector.get_embeddings(data_scaled)

    # Detect anomalies
    scores, anomalies = detector.detect_anomalies(data_scaled)

    # Print result
    results = pd.DataFrame({
        'customer_id': customer_ids,
        'anomaly_score': scores,
        'is_anomaly': anomalies
    })
    results = pd.concat([results, pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])], axis=1)
    print("Centroid:")
    print(detector.centroid)
    print("\nSample Results:")
    print(results.head())
    print(f"\nNumber of detected anomalies: {sum(anomalies)}")

    # Save embeddings to a text file
    output = np.column_stack([customer_ids, embeddings])
    output_file = 'customer_embeddings.txt'
    np.savetxt(output_dir / output_file, output, fmt='%s', delimiter=',')

    output_anomalies = results[results['is_anomaly'] == True]
    output_anomalies.to_csv(output_dir / 'anomalous_customers_embedding.csv')
    # Visualize anomaly score
    plt.hist(scores, bins=50)
    plt.title('Anomaly Score')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(output_dir / "Anomaly_Score.png")

    def visualize_anomalies(embeddings, anomalies, method='pca'):
        # Reduce dimensionality to 2D
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
            print(f"Explained variance ratio by PCA: {sum(reducer.explained_variance_ratio_):.3f}")
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        # Separate normal and anomalous points
        normal_points = embeddings_2d[~anomalies]
        anomaly_points = embeddings_2d[anomalies]

        plt.figure(figsize=(10, 8))
        # Plot normal points
        plt.scatter(normal_points[:, 0], normal_points[:, 1], 
                    c='blue', alpha=0.6, label='Normal', s=50)
        # Plot anomalies
        plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], 
                    c='red', alpha=0.8, label='Anomaly', s=100, marker='x')
        
        plt.title(f"Customer Transaction Embeddings ({method.upper()})", fontsize=14)
        plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
        plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f'visualize_anomalies_in_{method}')

    # Visualize anomalies using PCA (or change method to 'tsne')
    visualize_anomalies(embeddings, anomalies, method='pca')
