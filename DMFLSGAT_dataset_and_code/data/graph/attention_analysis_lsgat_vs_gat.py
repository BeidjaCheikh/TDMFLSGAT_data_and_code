# attention_analysis_lsgat_vs_gat.py

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------------------------------------------------------
# 0. IMPORTS DE TON PROJET EXISTANT
# ------------------------------------------------------------
# Adapter ce path à ton projet si besoin
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

from utils import (
    GraphUtils,
    set_seed,
)

# ============================================================
# 1. COUCHES GAT & LSGAT AVEC SAUVEGARDE D'ATTENTION
# ============================================================

class GraphAttentionLayer(nn.Module):
    """
    GAT simple, basé sur Velickovic et al. 2017
    -> Version adaptée pour stocker la matrice d'attention (self.last_attention)
    """
    def __init__(self, in_features=65, out_features=32, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # pour sauvegarder la dernière matrice d'attention (N x N)
        self.last_attention = None

    def forward(self, h, adj):
        # h: (N, in_features), adj: (N, N)
        Wh = torch.mm(h, self.W)  # (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)  # (N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        # on stocke la matrice d'attention (CPU pour analyse)
        self.last_attention = attention.detach().cpu()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_features)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' ({} -> {})'.format(
            self.in_features, self.out_features
        )


class LSGATLayer(nn.Module):
    """
    LSGAT : Layer-wise Self-Adaptive Graph Attention Network (Su et al., 2024)
    -> Version adaptée pour stocker la matrice d'attention (self.last_attention)
    """
    def __init__(self, in_features=65, out_features=32, dropout=0.5,
                 alpha=0.2, concat=True, beta=0.6, layer_id=1):
        super(LSGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.beta = beta
        self.layer_id = layer_id

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # pour sauvegarder la dernière matrice d'attention (N x N)
        self.last_attention = None

    def forward(self, h, adj):
        """
        h : (N, in_features)
        adj : (N, N)  (adjacence avec self-loop déjà gérée dans tes features)
        """
        device = adj.device
        N = adj.shape[0]

        Wh = torch.mm(h, self.W)   # (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)  # (N, N)

        # === Overlap-degree et scaling (Su et al.) ===
        A_power = torch.matrix_power(adj + torch.eye(N, device=device), self.layer_id)
        overlap_degree = A_power.sum(dim=0)  # (N,)

        tau = torch.quantile(overlap_degree, self.beta)
        scaled_overlap = overlap_degree / (tau + 1e-8)
        scaling_score = torch.where(
            scaled_overlap <= 1,
            torch.ones_like(scaled_overlap),
            1.0 / scaled_overlap
        )  # (N,)

        e = e * scaling_score.unsqueeze(0)  # (N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(self.leakyrelu(attention), dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        # stocker la matrice d'attention
        self.last_attention = attention.detach().cpu()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T  # (N, N)
        return e

    def __repr__(self):
        return self.__class__.__name__ + f' ({self.in_features} -> {self.out_features})'


# ============================================================
# 2. MODELES GRAPHIQUES (LSGAT vs GAT)
# ============================================================

class GraphModelLSGAT(nn.Module):
    """
    Modèle graphe avec 5 couches LSGAT (4 têtes) + MLP + sigmoïde
    """
    def __init__(self, num_layers=5, num_heads=4, in_features=65,
                 out_features=32, dropout=0.5, alpha=0.2, beta=0.6,
                 max_nodes=100):
        super(GraphModelLSGAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_features = out_features
        self.max_nodes = max_nodes

        self.lsgat_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_features if i == 0 else out_features * num_heads
            self.lsgat_layers.append(
                nn.ModuleList([
                    LSGATLayer(
                        in_features=input_dim,
                        out_features=out_features,
                        dropout=dropout,
                        alpha=alpha,
                        concat=True,
                        beta=beta,
                        layer_id=i + 1
                    )
                    for _ in range(num_heads)
                ])
            )

        self.proj = nn.Sequential(
            nn.Linear(out_features * num_heads * max_nodes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, X, A):
        """
        X : (batch, N, F)
        A : (batch, N, N)
        """
        batch_size, N, _ = X.shape
        x = X
        for layer in self.lsgat_layers:
            head_outputs = []
            for lsgat in layer:
                temp = []
                for i in range(batch_size):
                    temp.append(lsgat(x[i], A[i]))
                temp = torch.stack(temp, dim=0)  # (batch, N, out_features)
                head_outputs.append(temp)
            x = torch.cat(head_outputs, dim=2)  # (batch, N, out_features * num_heads)

        out = x.view(batch_size, -1)
        out = self.proj(out)
        out = torch.sigmoid(out).squeeze(-1)  # (batch,)
        return out

    def get_last_head_attention(self):
        """
        Retourne la matrice d'attention (N x N) du
        dernier LSGAT d'une tête (par ex. head 0).
        """
        last_layer = self.lsgat_layers[-1][0]
        return last_layer.last_attention  # (N, N) sur CPU


class GraphModelGAT(nn.Module):
    """
    Modèle graphe avec 5 couches GAT (4 têtes) + MLP + sigmoïde
    """
    def __init__(self, num_layers=5, num_heads=4, in_features=65,
                 out_features=32, dropout=0.5, alpha=0.2, max_nodes=100):
        super(GraphModelGAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_features = out_features
        self.max_nodes = max_nodes

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_features if i == 0 else out_features * num_heads
            self.gat_layers.append(
                nn.ModuleList([
                    GraphAttentionLayer(
                        in_features=input_dim,
                        out_features=out_features,
                        dropout=dropout,
                        alpha=alpha,
                        concat=True
                    )
                    for _ in range(num_heads)
                ])
            )

        self.proj = nn.Sequential(
            nn.Linear(out_features * num_heads * max_nodes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, X, A):
        batch_size, N, _ = X.shape
        x = X
        for layer in self.gat_layers:
            head_outputs = []
            for gat in layer:
                temp = []
                for i in range(batch_size):
                    temp.append(gat(x[i], A[i]))
                temp = torch.stack(temp, dim=0)
                head_outputs.append(temp)
            x = torch.cat(head_outputs, dim=2)

        out = x.view(batch_size, -1)
        out = self.proj(out)
        out = torch.sigmoid(out).squeeze(-1)
        return out

    def get_last_head_attention(self):
        last_layer = self.gat_layers[-1][0]
        return last_layer.last_attention  # (N, N) sur CPU


# ============================================================
# 3. DATASET GRAPHIQUE POUR ENTRAINER LES MODELES
# ============================================================

class GraphOnlyDataset(Dataset):
    """
    Dataset PyTorch pour (X, A, label) uniquement.
    On se concentre sur la branche graphe pour analyser LSGAT vs GAT.
    """
    def __init__(self, X, A, labels):
        self.X = X
        self.A = A
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.X[idx],       # (100, 65)
            self.A[idx],       # (100, 100)
            self.labels[idx]   # scalaire 0/1
        )


# ============================================================
# 4. CHARGEMENT DU DATASET INTERNE
# ============================================================

def load_internal_graph_dataset(csv_path):
    """
    Utilise GraphUtils de ton projet pour reconstruire X, A à partir de SMILES.
    """
    data = pd.read_csv(csv_path)
    smiles = list(data['smiles'])
    labels = data['labels'].values.astype(np.float32)

    gutils = GraphUtils()
    X, A = gutils.preprocess_smile(smiles)   # X: (N, 100, 65), A: (N, 100, 100)

    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    labels = torch.FloatTensor(labels)

    return X, A, labels


# ============================================================
# 5. ENTRAINEMENT SIMPLE DES MODELES
# ============================================================

def train_graph_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.to(device)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for X_b, A_b, y_b in train_loader:
            X_b = X_b.to(device)
            A_b = A_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            preds = model(X_b, A_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, A_b, y_b in val_loader:
                X_b = X_b.to(device)
                A_b = A_b.to(device)
                y_b = y_b.to(device)
                preds = model(X_b, A_b)
                loss = criterion(preds, y_b)
                val_losses.append(loss.item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss = {mean_train:.4f} | "
              f"Val Loss = {mean_val:.4f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# 6. ANALYSE D'ATTENTION : FRACTION SUR LES HUBS
# ============================================================

def compute_attention_stats(model, dataset, device, max_samples=None):
    """
    Pour un modèle (GAT ou LSGAT), calcule :

      - frac_high_list : liste, pour chaque molécule,
        fraction de la masse totale d'attention qui tombe
        sur les atomes de haut degré (top 10 % en degré).

      - example_attention : une matrice d'attention (N x N) d'exemple
      - example_degree    : degrés des nœuds pour cette molécule

    max_samples = None -> utilise tout le dataset.
    """
    model.to(device)
    model.eval()

    frac_high_list = []
    example_attention = None
    example_degree = None

    with torch.no_grad():
        if max_samples is None:
            n_samples = len(dataset)
        else:
            n_samples = min(max_samples, len(dataset))

        for idx in range(n_samples):
            X_i, A_i, y_i = dataset[idx]
            X_i = X_i.unsqueeze(0).to(device)   # (1, N, F)
            A_i = A_i.unsqueeze(0).to(device)   # (1, N, N)

            # forward pour remplir last_attention
            _ = model(X_i, A_i)

            # matrice d'attention (N, N)
            att = model.get_last_head_attention().numpy()
            # matrice d'adjacence (N, N)
            A_cpu = A_i.squeeze(0).cpu().numpy()

            # degré (on n'utilise que les vrais atomes : degré > 0)
            degree = A_cpu.sum(axis=0)
            valid_mask = degree > 0
            if valid_mask.sum() == 0:
                continue

            valid_degrees = degree[valid_mask]
            cutoff = np.percentile(valid_degrees, 90)
            high_mask = (degree >= cutoff) & (degree > 0)

            if high_mask.sum() == 0:
                continue

            # somme totale de l'attention
            total_att = att.sum() + 1e-12
            # somme de l'attention qui arrive sur les hubs (colonnes high_mask)
            att_high = att[:, high_mask].sum()
            frac_high = att_high / total_att   # fraction entre 0 et 1

            frac_high_list.append(frac_high)

            # garder un exemple pour les heatmaps
            if example_attention is None:
                example_attention = att
                example_degree = degree

    overall_mean = float(np.mean(frac_high_list)) if len(frac_high_list) > 0 else 0.0
    return overall_mean, frac_high_list, example_attention, example_degree


# ============================================================
# 7. MAIN : TOUT LANCER (TRAIN + ANALYSE + FIGURES)
# ============================================================

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Charger ton dataset interne (le même que DMFGAM/DMFLSGAT)
    csv_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/Smiles.csv'
    X, A, labels = load_internal_graph_dataset(csv_path)

    # petit split train/val pour entraîner les modèles graphe
    N = len(labels)
    indices = np.arange(N)
    np.random.shuffle(indices)

    split = int(0.8 * N)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = GraphOnlyDataset(X[train_idx], A[train_idx], labels[train_idx])
    val_dataset   = GraphOnlyDataset(X[val_idx], A[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=100, shuffle=False, drop_last=True)

    # 2) Instancier les deux modèles
    model_lsgat = GraphModelLSGAT(
        num_layers=5,
        num_heads=4,
        in_features=65,
        out_features=32,
        dropout=0.3,
        alpha=0.2,
        beta=0.6,
        max_nodes=100
    )

    model_gat = GraphModelGAT(
        num_layers=5,
        num_heads=4,
        in_features=65,
        out_features=32,
        dropout=0.3,
        alpha=0.2,
        max_nodes=100
    )

    # 3) Entraîner les modèles rapidement (20 epochs chacun)
    print("\n=== Training LSGAT model ===")
    model_lsgat = train_graph_model(model_lsgat, train_loader, val_loader,
                                    device=device, num_epochs=5, lr=1e-4)

    print("\n=== Training GAT model ===")
    model_gat = train_graph_model(model_gat, train_loader, val_loader,
                                  device=device, num_epochs=5, lr=1e-4)

    # 4) Analyse d'attention sur TOUT le val_dataset
    print("\n=== Attention analysis: LSGAT ===")
    mean_frac_lsgat, frac_list_lsgat, att_ex_lsgat, deg_ex = compute_attention_stats(
        model_lsgat, val_dataset, device=device, max_samples=None
    )
    print(f"Mean fraction of attention on high-degree atoms (LSGAT) = {mean_frac_lsgat:.3f}")

    print("\n=== Attention analysis: GAT ===")
    mean_frac_gat, frac_list_gat, att_ex_gat, _ = compute_attention_stats(
        model_gat, val_dataset, device=device, max_samples=None
    )
    print(f"Mean fraction of attention on high-degree atoms (GAT) = {mean_frac_gat:.3f}")

    # 5) FIGURES POUR LE MANUSCRIT
    # --------------------------------------------------------
    # Heatmaps attention GAT vs LSGAT sur un exemple
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(att_ex_gat)
    plt.title("Standard GAT – Atom–Atom Attention (Example)")
    plt.xlabel("Target atom j")
    plt.ylabel("Source atom i")
    plt.tight_layout()
    plt.savefig("gat_attention_example.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(att_ex_lsgat)
    plt.title("LSGAT – Atom–Atom Attention (Example)")
    plt.xlabel("Target atom j")
    plt.ylabel("Source atom i")
    plt.tight_layout()
    plt.savefig("lsgat_attention_example.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Boxplot des fractions d'attention sur les nœuds de haut degré
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))
    data_box = [
        frac_list_gat,
        frac_list_lsgat
    ]
    plt.boxplot(data_box, labels=["GAT", "LSGAT"])
    plt.ylabel("Fraction of total attention on high-degree atoms")
    plt.title("Attention redistribution on hub atoms")
    plt.tight_layout()
    plt.savefig("attention_high_degree_fraction_boxplot.png", dpi=300)
    plt.close()

    print("\nFigures saved:")
    print(" - gat_attention_example.png")
    print(" - lsgat_attention_example.png")
    print(" - attention_high_degree_fraction_boxplot.png")
