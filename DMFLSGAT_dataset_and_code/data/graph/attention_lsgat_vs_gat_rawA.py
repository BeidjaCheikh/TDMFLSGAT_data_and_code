# attention_lsgat_vs_gat_rawA.py

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger

# désactiver les warnings RDKit
RDLogger.DisableLog('rdApp.warning')

# ==== CONFIG =====
CSV_PATH = r"/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/Smiles.csv"
MAX_ATOMS = 100
ATOM_FEATS = 65
SEED = 42
# =================


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 0. Construction du graphe avec A BRUTE (A + I, PAS normalize_adj)
# ============================================================

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def get_ring_info(atom):
    ring_info_feature = []
    for i in range(3, 9):
        ring_info_feature.append(1 if atom.IsInRingSize(i) else 0)
    return ring_info_feature


def atom_feature(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I',
                'B', 'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                'Ti', 'Zn', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr',
                'Pt', 'Hg', 'Pb'
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        + [atom.GetIsAromatic()]
        + get_ring_info(atom)
    )


def normalize_rows(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1, where=rowsum != 0).flatten()
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv @ mx


def smiles_to_graph(smiles_list):
    features = []
    adjs = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        A_tmp = Chem.rdmolops.GetAdjacencyMatrix(mol)  # (natoms, natoms)

        natoms = mol.GetNumAtoms()
        # --- features (MAX_ATOMS x 65), normalisation ligne ---
        feat = np.zeros((MAX_ATOMS, ATOM_FEATS), dtype=float)
        tmp_feat = [atom_feature(a) for a in mol.GetAtoms()]
        tmp_feat = np.array(tmp_feat, dtype=float)
        tmp_feat = normalize_rows(tmp_feat)
        feat[:natoms, :] = tmp_feat

        # --- adjacency brute + self-loop, PAS normalize_adj ---
        A = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=float)
        A[:natoms, :natoms] = A_tmp + np.eye(natoms, dtype=float)

        features.append(feat)
        adjs.append(A)

    return np.asarray(features), np.asarray(adjs)


def load_internal_graph_dataset(csv_path):
    data = pd.read_csv(csv_path)
    smiles = list(data["smiles"])
    labels = data["labels"].values.astype(np.float32)

    X, A = smiles_to_graph(smiles)
    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    labels = torch.FloatTensor(labels)

    return X, A, labels


# ============================================================
# 1. Couches GAT & LSGAT (avec attention stockée)
# ============================================================

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features=65, out_features=32, dropout=0.5, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.last_attention = None

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)  # (N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        self.last_attention = attention.detach().cpu()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class LSGATLayer(nn.Module):
    def __init__(
        self,
        in_features=65,
        out_features=32,
        dropout=0.5,
        alpha=0.2,
        concat=True,
        beta=0.6,
        layer_id=1,
    ):
        super().__init__()
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

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.last_attention = None

    def forward(self, h, adj):
        device = adj.device
        N = adj.shape[0]

        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        # Overlap-degree sur A brute (+I déjà inclus)
        A_power = torch.matrix_power(adj, self.layer_id)
        overlap_degree = A_power.sum(dim=0)  # (N,)

        tau = torch.quantile(overlap_degree, self.beta)
        scaled = overlap_degree / (tau + 1e-8)
        scaling_score = torch.where(
            scaled <= 1.0,
            torch.ones_like(scaled),
            1.0 / scaled,
        )  # (N,)

        # ✅ scaling AVANT softmax
        e = e * scaling_score.unsqueeze(0)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(self.leakyrelu(attention), dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        self.last_attention = attention.detach().cpu()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        return Wh1 + Wh2.T


# ============================================================
# 2. Modèles graphe + Dataset
# ============================================================

class GraphModelLSGAT(nn.Module):
    def __init__(
        self,
        num_layers=5,
        num_heads=4,
        in_features=65,
        out_features=32,
        dropout=0.5,
        alpha=0.2,
        beta=0.6,
        max_nodes=100,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_features = out_features
        self.max_nodes = max_nodes

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_features if i == 0 else out_features * num_heads
            self.layers.append(
                nn.ModuleList(
                    [
                        LSGATLayer(
                            in_features=input_dim,
                            out_features=out_features,
                            dropout=dropout,
                            alpha=alpha,
                            concat=True,
                            beta=beta,
                            layer_id=i + 1,
                        )
                        for _ in range(num_heads)
                    ]
                )
            )

        self.proj = nn.Sequential(
            nn.Linear(out_features * num_heads * max_nodes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, X, A):
        b, N, _ = X.shape
        x = X
        for layer in self.layers:
            heads = []
            for l in layer:
                out = []
                for i in range(b):
                    out.append(l(x[i], A[i]))
                out = torch.stack(out, dim=0)
                heads.append(out)
            x = torch.cat(heads, dim=2)

        out = x.view(b, -1)
        out = self.proj(out)
        return torch.sigmoid(out).squeeze(-1)

    def get_last_head_attention(self):
        return self.layers[-1][0].last_attention


class GraphModelGAT(nn.Module):
    def __init__(
        self,
        num_layers=5,
        num_heads=4,
        in_features=65,
        out_features=32,
        dropout=0.5,
        alpha=0.2,
        max_nodes=100,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_features = out_features
        self.max_nodes = max_nodes

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_features if i == 0 else out_features * num_heads
            self.layers.append(
                nn.ModuleList(
                    [
                        GraphAttentionLayer(
                            in_features=input_dim,
                            out_features=out_features,
                            dropout=dropout,
                            alpha=alpha,
                            concat=True,
                        )
                        for _ in range(num_heads)
                    ]
                )
            )

        self.proj = nn.Sequential(
            nn.Linear(out_features * num_heads * max_nodes, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, X, A):
        b, N, _ = X.shape
        x = X
        for layer in self.layers:
            heads = []
            for l in layer:
                out = []
                for i in range(b):
                    out.append(l(x[i], A[i]))
                out = torch.stack(out, dim=0)
                heads.append(out)
            x = torch.cat(heads, dim=2)

        out = x.view(b, -1)
        out = self.proj(out)
        return torch.sigmoid(out).squeeze(-1)

    def get_last_head_attention(self):
        return self.layers[-1][0].last_attention


class GraphOnlyDataset(Dataset):
    def __init__(self, X, A, y):
        self.X = X
        self.A = A
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.A[i], self.y[i]


# ============================================================
# 3. Entraînement + analyse d'attention
# ============================================================

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    crit = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)

    best = None
    best_val = 1e9

    for ep in range(epochs):
        model.train()
        losses = []
        for Xb, Ab, yb in train_loader:
            Xb, Ab, yb = Xb.to(device), Ab.to(device), yb.to(device)
            optim.zero_grad()
            p = model(Xb, Ab)
            loss = crit(p, yb)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        model.eval()
        vloss = []
        with torch.no_grad():
            for Xb, Ab, yb in val_loader:
                Xb, Ab, yb = Xb.to(device), Ab.to(device), yb.to(device)
                p = model(Xb, Ab)
                loss = crit(p, yb)
                vloss.append(loss.item())
        print(
            f"[Epoch {ep+1}/{epochs}] train={np.mean(losses):.4f} val={np.mean(vloss):.4f}"
        )
        if np.mean(vloss) < best_val:
            best_val = np.mean(vloss)
            best = model.state_dict()
    if best is not None:
        model.load_state_dict(best)
    return model


def compute_attention_stats(model, dataset, device, max_samples=None):
    model.to(device)
    model.eval()
    frac_list = []
    example_att = None
    example_deg = None

    with torch.no_grad():
        if max_samples is None:
            n = len(dataset)
        else:
            n = min(max_samples, len(dataset))

        for idx in range(n):
            X_i, A_i, y_i = dataset[idx]
            X_i = X_i.unsqueeze(0).to(device)
            A_i = A_i.unsqueeze(0).to(device)

            _ = model(X_i, A_i)
            att = model.get_last_head_attention().numpy()
            A_cpu = A_i.squeeze(0).cpu().numpy()

            degree = A_cpu.sum(axis=0)
            valid = degree > 0
            if valid.sum() == 0:
                continue

            valid_degrees = degree[valid]
            cutoff = np.percentile(valid_degrees, 90)
            high = (degree >= cutoff) & (degree > 0)
            if high.sum() == 0:
                continue

            tot = att.sum() + 1e-12
            att_high = att[:, high].sum()
            frac = att_high / tot
            frac_list.append(frac)

            if example_att is None:
                example_att = att
                example_deg = degree

    mean_frac = float(np.mean(frac_list)) if frac_list else 0.0
    return mean_frac, frac_list, example_att, example_deg


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # chargement dataset (A brute + I)
    X, A, y = load_internal_graph_dataset(CSV_PATH)

    N = len(y)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    tr_idx, val_idx = idx[:split], idx[split:]

    train_ds = GraphOnlyDataset(X[tr_idx], A[tr_idx], y[tr_idx])
    val_ds = GraphOnlyDataset(X[val_idx], A[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # modèles
    model_lsgat = GraphModelLSGAT(beta=0.6, max_nodes=MAX_ATOMS)
    model_gat = GraphModelGAT(max_nodes=MAX_ATOMS)

    print("\n=== Training LSGAT ===")
    model_lsgat = train_model(model_lsgat, train_loader, val_loader, device)

    print("\n=== Training GAT ===")
    model_gat = train_model(model_gat, train_loader, val_loader, device)

    # analyse
    print("\n=== Attention analysis: LSGAT ===")
    mean_l, frac_l, att_l, deg_l = compute_attention_stats(
        model_lsgat, val_ds, device, max_samples=None
    )
    print(f"Mean fraction on high-degree atoms (LSGAT) = {mean_l:.3f}")

    print("\n=== Attention analysis: GAT ===")
    mean_g, frac_g, att_g, _ = compute_attention_stats(
        model_gat, val_ds, device, max_samples=None
    )
    print(f"Mean fraction on high-degree atoms (GAT) = {mean_g:.3f}")

    # ==== figures ====
    # 1) heatmaps recadrées sur les vrais atomes
    if att_g is not None and att_l is not None and deg_l is not None:
        mask = deg_l > 0
        n_real = int(mask.sum())
        att_g_crop = att_g[:n_real, :n_real]
        att_l_crop = att_l[:n_real, :n_real]

        plt.figure(figsize=(5, 4))
        sns.heatmap(att_g_crop)
        plt.title("Standard GAT – Atom–Atom Attention (Example)")
        plt.xlabel("Target atom j")
        plt.ylabel("Source atom i")
        plt.tight_layout()
        plt.savefig("gat_attention_example_cropped.png", dpi=300)
        plt.close()

        plt.figure(figsize=(5, 4))
        sns.heatmap(att_l_crop)
        plt.title("LSGAT – Atom–Atom Attention (Example)")
        plt.xlabel("Target atom j")
        plt.ylabel("Source atom i")
        plt.tight_layout()
        plt.savefig("lsgat_attention_example_cropped.png", dpi=300)
        plt.close()

    # 2) boxplot fractions
    plt.figure(figsize=(6, 5))
    plt.boxplot([frac_g, frac_l], labels=["GAT", "LSGAT"])
    plt.ylabel("Fraction of total attention on high-degree atoms")
    plt.title("Attention redistribution on hub atoms")
    plt.tight_layout()
    plt.savefig("attention_high_degree_fraction_boxplot.png", dpi=300)
    plt.close()

    print("\nFigures saved:")
    print(" - gat_attention_example_cropped.png")
    print(" - lsgat_attention_example_cropped.png")
    print(" - attention_high_degree_fraction_boxplot.png")
