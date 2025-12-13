import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from argparse import Namespace
from copy import deepcopy
from hyperopt import fmin, tpe, hp, Trials

# Ajout du chemin pour charger les modules spécifiques
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data')
from graph.layer import GraphAttentionLayer
from graph.utils import load_data_with_smiles, set_seed, load_fp
from utils import SmilesUtils

##############################################
# 1. Définition des modèles (comme dans model.py)
##############################################

class SmilesTransformer(nn.Module):
    """Encodeur Transformer pour les séquences SMILES."""
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=256, num_layers=6, max_len=79, dropout=0.1):
        super(SmilesTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        pe = self._generate_positional_encoding(max_len, embed_dim)
        self.register_buffer("positional_encoding", pe)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def _generate_positional_encoding(self, max_len, embed_dim):
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, embed_dim]

    def forward(self, smiles_tokens):
        x = self.embedding(smiles_tokens) + self.positional_encoding[:, :smiles_tokens.size(1), :]
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return x

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.randn(65, 65), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(65), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        X, A = inputs
        xw = torch.matmul(X, self.weight)
        out = torch.matmul(A, xw)
        out += self.bias
        out = self.relu(out)
        return out, A

class GraphModel(nn.Module):
    def __init__(self, num_gcn_layers=3, num_gat_heads=4):
        super(GraphModel, self).__init__()
        self.num_head = num_gat_heads
        self.att = GraphAttentionLayer()
        self.proj = nn.Sequential(
            nn.Linear(32 * self.num_head * 100, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, X, A):
        features = []
        for i in range(X.shape[0]):
            feature_temp = []
            x, a = X[i], A[i]
            for _ in range(self.num_head):
                ax = self.att(x, a)
                feature_temp.append(ax)
            feature_temp = torch.cat(feature_temp, dim=1)
            features.append(feature_temp)
        out = torch.stack(features, dim=0)
        out = out.view(out.size(0), -1)
        out = self.proj(out)
        return out

class FpModel(nn.Module):
    def __init__(self):
        super(FpModel, self).__init__()
        self.fp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fp1 = nn.Sequential(
            nn.Linear(881, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fp3 = nn.Sequential(
            nn.Linear(780, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fp4 = nn.Sequential(
            nn.Linear(4860, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x, x1, x2, x3, x4):
        x = self.fp(x)
        x1 = self.fp1(x1)
        x2 = self.fp2(x2)
        x3 = self.fp3(x3)
        x4 = self.fp4(x4)
        return x, x1, x2, x3, x4

class MyModel(nn.Module):
    """Modèle complet combinant SMILES Transformer, Graph Model et FpModel."""
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=256, num_layers=6,
                 dropout=0.1, num_gcn_layers=3, num_gat_heads=4):
        super(MyModel, self).__init__()
        self.smiles_transformer = SmilesTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.graph_model = GraphModel(num_gcn_layers=num_gcn_layers, num_gat_heads=num_gat_heads)
        self.fp_model = FpModel()
        # Dimension combinée: embed_dim (Transformer) + 128 (GraphModel) + 640 (FpModel)
        combined_dim = embed_dim + 128 + 640
        self.proj = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, 1)
        self.active = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, smiles_tokens, f, f1, f2, f3, f4, X, A, label):
        smiles_features = self.smiles_transformer(smiles_tokens)
        graph_features = self.graph_model(X, A)
        fp_features = self.fp_model(f, f1, f2, f3, f4)
        fp_concat = torch.cat(fp_features, dim=1)
        combined = torch.cat((smiles_features, graph_features, fp_concat), dim=1)
        x = self.proj(combined)
        x = self.fc(x)
        x = self.active(x).squeeze(-1)
        loss = self.loss_fn(x, label)
        return x, loss

##############################################
# 2. Dataset et fonction d'entraînement
##############################################

class MyDataset(Dataset):
    def __init__(self, smiles_tokens, f, f1, f2, f3, f4, X, A, label):
        self.smiles_tokens = smiles_tokens
        self.f = f
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.X = X
        self.A = A
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return (self.smiles_tokens[index],
                self.f[index],
                self.f1[index],
                self.f2[index],
                self.f3[index],
                self.f4[index],
                self.X[index],
                self.A[index],
                self.label[index])

def training(args, log):
    """
    Entraîne le modèle MyModel sur l'ensemble d'entraînement et l'évalue sur la validation.
    Calcule l'accuracy et l'AUC, puis retourne la métrique combinée (moyenne de ACC et AUC).
    """
    set_seed(42)
    # Chargement des données et entraînement du tokenizer
    smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/Smiles.csv'
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()
    smiles_utils.train_tokenizer(smiles_list)
    X, A, mogen_fp, labels, smiles_tokens = load_data_with_smiles(path, smiles_utils)

    # Chargement des fingerprints
    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/PubchemFP881.csv'
    Topological_torsion_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/topological_torsion_fingerprints.csv'
    APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/APC2D780.csv'
    KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/KR.csv'
    fp1 = torch.FloatTensor(load_fp(PubchemFP881_path))
    fp2 = torch.FloatTensor(load_fp(Topological_torsion_path))
    fp3 = torch.FloatTensor(load_fp(APC2D780_path))
    fp4 = torch.FloatTensor(load_fp(KR_path))

    # Conversion en tenseurs
    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    smiles_tokens = torch.LongTensor(smiles_tokens)
    labels = torch.FloatTensor(labels)

    # Création des datasets (train/validation)
    train_dataset = MyDataset(
        smiles_tokens[0:8284], mogen_fp[0:8284], fp1[0:8284], fp2[0:8284], fp3[0:8284], fp4[0:8284],
        X[0:8284], A[0:8284], labels[0:8284]
    )
    valid_dataset = MyDataset(
        smiles_tokens[8284:], mogen_fp[8284:], fp1[8284:], fp2[8284:], fp3[8284:], fp4[8284:],
        X[8284:], A[8284:], labels[8284:]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Initialisation du modèle avec dropout, num_heads, num_layers fixés ou optimisés
    model = MyModel(
        vocab_size=5000,
        embed_dim=128,  # Fixé
        ff_dim=256,     # Fixé
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_gcn_layers=args.num_gcn_layers,
        num_gat_heads=args.num_gat_heads
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Taux d'apprentissage fixé ici (par exemple 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

    num_epochs = args.num_epochs

    log.write("Début de l'entraînement\n")
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            (smiles_tokens_b, fp, fp1_b, fp2_b, fp3_b, fp4_b,
             X_batch, A_batch, label) = batch

            smiles_tokens_b = smiles_tokens_b.to(device)
            fp = fp.to(device)
            fp1_b = fp1_b.to(device)
            fp2_b = fp2_b.to(device)
            fp3_b = fp3_b.to(device)
            fp4_b = fp4_b.to(device)
            X_batch = X_batch.to(device)
            A_batch = A_batch.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits, loss = model(smiles_tokens_b, fp, fp1_b, fp2_b, fp3_b, fp4_b, X_batch, A_batch, label)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        log.write(f"Epoch {epoch} - Loss: {loss.item():.4f}\n")

    # Évaluation sur validation
    model.eval()
    val_pred = []
    val_true = []
    with torch.no_grad():
        for batch in valid_loader:
            (smiles_tokens_b, fp, fp1_b, fp2_b, fp3_b, fp4_b,
             X_batch, A_batch, label) = batch

            smiles_tokens_b = smiles_tokens_b.to(device)
            fp = fp.to(device)
            fp1_b = fp1_b.to(device)
            fp2_b = fp2_b.to(device)
            fp3_b = fp3_b.to(device)
            fp4_b = fp4_b.to(device)
            X_batch = X_batch.to(device)
            A_batch = A_batch.to(device)
            label = label.to(device)

            logits, _ = model(smiles_tokens_b, fp, fp1_b, fp2_b, fp3_b, fp4_b, X_batch, A_batch, label)
            val_pred.extend(logits.detach().cpu().numpy())
            val_true.extend(label.detach().cpu().numpy())

    val_pred_bin = np.array(val_pred) > 0.5
    acc = accuracy_score(np.array(val_true), val_pred_bin)
    try:
        auc = roc_auc_score(np.array(val_true), np.array(val_pred))
    except:
        auc = 0.0

    log.write(f"Validation Accuracy: {acc:.4f}\n")
    log.write(f"Validation AUC: {auc:.4f}\n")

    combined_metric = (acc + auc) / 2.0
    return combined_metric, 0.0

##############################################
# 3. Hyperparameter Optimization
##############################################

class SimpleLogger:
    def __init__(self, log_name, log_dir):
        self.log_file = os.path.join(log_dir, log_name + ".txt")
    def write(self, message):
        print(message.strip())
        with open(self.log_file, "a") as f:
            f.write(message)

def set_log(log_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SimpleLogger(log_name, log_dir)

def set_hyper_argument():
    args = Namespace()
    # Paramètres fixés
    args.embed_dim = 128
    args.ff_dim = 256
    # Paramètres par défaut (qui pourront être surchargés par Hyperopt)
    args.num_heads = 8
    args.num_layers = 6
    args.dropout = 0.1
    args.num_gcn_layers = 3
    args.num_gat_heads = 4
    args.num_epochs = 5
    args.batch_size = 64
    # Fixation du taux d'apprentissage
    args.lr = 1e-4
    # Chemins et autres paramètres
    args.save_path = "./checkpoints"
    args.log_path = "./logs"
    args.dataset_type = "classification"
    args.metric = "combined_metric"
    args.search_num = 10000  # Nombre d'évaluations par hyperopt
    args.search_now = 0
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    return args

# Espace de recherche : on optimise batch_size, num_layers, num_gcn_layers, num_gat_heads, num_epochs,
# ainsi que num_heads (Transformer) et dropout.
space = {
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'num_layers': hp.quniform('num_layers', low=3, high=8, q=1),
    'num_gcn_layers': hp.quniform('num_gcn_layers', low=2, high=5, q=1),
    'num_gat_heads': hp.quniform('num_gat_heads', low=2, high=8, q=1),
    'num_epochs': hp.quniform('num_epochs', low=3, high=10, q=1),
    'num_heads': hp.choice('num_heads', [2, 4, 8]),  # diviseurs de 128
    'dropout': hp.quniform('dropout', low=0.1, high=0.5, q=0.05)
}

def objective(hyperparams):
    # Conversion
    hyperparams['batch_size'] = int(hyperparams['batch_size'])
    hyperparams['num_layers'] = int(hyperparams['num_layers'])
    hyperparams['num_gcn_layers'] = int(hyperparams['num_gcn_layers'])
    hyperparams['num_gat_heads'] = int(hyperparams['num_gat_heads'])
    hyperparams['num_epochs'] = int(hyperparams['num_epochs'])
    hyperparams['dropout'] = float(hyperparams['dropout'])
    # num_heads est déjà choisi parmi [2,4,8] via hp.choice

    args = set_hyper_argument()
    for key, value in hyperparams.items():
        setattr(args, key, value)

    # Création d'un nom de dossier spécifique pour cette configuration
    dir_name = (f"batch{args.batch_size}_layers{args.num_layers}_GCN{args.num_gcn_layers}_"
                f"GAT{args.num_gat_heads}_epochs{args.num_epochs}_heads{args.num_heads}_"
                f"drop{args.dropout:.2f}")
    args.save_path = os.path.join(args.save_path, dir_name)
    log_name = "train_" + dir_name
    log = set_log(log_name, args.log_path)

    metric, std = training(args, log)

    result_path = os.path.join(args.log_path, "hyper_para_result.txt")
    with open(result_path, "a") as file:
        file.write(f"Hyperparams: {hyperparams}\n")
        file.write(f"Result ({args.metric}): {metric:.4f} +/- {std:.4f}\n")

    args.search_now += 1
    # On minimise l'opposé de la métrique (car on maximise la métrique combinée)
    return -metric

def hyper_search(args):
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.search_num,
        trials=trials
    )
    result_path = os.path.join(args.log_path, "hyper_para_result.txt")
    with open(result_path, "a") as file:
        file.write("Best Hyperparameters:\n")
        file.write(str(best) + "\n")
    print("Meilleure configuration:", best)

    # Réentraîner le modèle avec la meilleure configuration trouvée
    best_args = set_hyper_argument()
    best_args.batch_size = [16, 32, 64, 128][best['batch_size']]  # hp.choice retourne un indice
    best_args.num_layers = int(best['num_layers'])
    best_args.num_gcn_layers = int(best['num_gcn_layers'])
    best_args.num_gat_heads = int(best['num_gat_heads'])
    best_args.num_epochs = int(best['num_epochs'])
    best_args.num_heads = [2, 4, 8][best['num_heads']]
    best_args.dropout = float(best['dropout'])

    final_log = set_log("best_model", best_args.log_path)
    metric, std = training(best_args, final_log)
    print("Réentraîner final:")
    print(f"Combined metric (ACC/AUC) = {metric:.4f}")

##############################################
# 4. Exécution principale
##############################################
if __name__ == '__main__':
    args = set_hyper_argument()
    hyper_search(args)
