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
from sklearn.model_selection import KFold
from argparse import Namespace
from copy import deepcopy
from hyperopt import fmin, tpe, hp, Trials

# Ajout du chemin pour charger les modules spécifiques
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data')
from DMFLSGAT_dataset_and_code.data.graph.LSGATLayer import GraphAttentionLayer
from DMFLSGAT_dataset_and_code.data.graph.DMFLSGATUtils import load_data_with_smiles, set_seed, load_fp
from DMFLSGAT_dataset_and_code.data.graph.DMFLSGATUtils import SmilesUtils

##############################################
# 1. Définition des modèles (comme dans model.py)
##############################################

class SmilesTransformer(nn.Module):
    """Encodeur Transformer pour les séquences SMILES."""
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=256, num_layers=6, max_len=79, dropout=0.1):
        super(SmilesTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Créer l'encodage positionnel et l'enregistrer comme buffer
        pe = self._generate_positional_encoding(max_len, embed_dim)
        self.register_buffer("positional_encoding", pe)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def _generate_positional_encoding(self, max_len, embed_dim):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

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
    def __init__(self):
        super(GraphModel, self).__init__()
        self.num_head = 4
        self.att = GraphAttentionLayer()
        self.proj = nn.Sequential(
            nn.Linear(32*4*100, 1024),
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
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=256, num_layers=6, dropout=0.1):
        super(MyModel, self).__init__()
        self.smiles_transformer = SmilesTransformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=dropout)
        self.graph_model = GraphModel()
        self.fp_model = FpModel()
        # Dimension combinée : embed_dim (Transformer) + 128 (Graph) + 640 (FpModel)
        combined_dim = embed_dim + 128 + 640  # soit embed_dim + 768
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
# 2. Dataset et fonction d'entraînement réelle
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
        return (self.smiles_tokens[index], self.f[index], self.f1[index], self.f2[index],
                self.f3[index], self.f4[index], self.X[index], self.A[index], self.label[index])

def training(args, log):
    """
    Entraîne le modèle MyModel sur l'ensemble d'entraînement et
    évalue sur le jeu de validation en calculant à la fois l'accuracy et l'AUC.
    La métrique retournée est la moyenne de l'accuracy et de l'AUC.
    """
    set_seed(42)
    # Chargement et préparation des données
    smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/Smiles.csv'
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()
    smiles_utils.train_tokenizer(smiles_list)
    X, A, mogen_fp, labels, smiles_tokens = load_data_with_smiles(path, smiles_utils)
    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/PubchemFP881.csv'
    Topological_torsion_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/topological_torsion_fingerprints.csv'
    APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/APC2D780.csv'
    KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/KR.csv'
    fp1 = torch.FloatTensor(load_fp(PubchemFP881_path))
    fp2 = torch.FloatTensor(load_fp(Topological_torsion_path))
    fp3 = torch.FloatTensor(load_fp(APC2D780_path))
    fp4 = torch.FloatTensor(load_fp(KR_path))
    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    smiles_tokens = torch.LongTensor(smiles_tokens)
    labels = torch.FloatTensor(labels)

    train_dataset = MyDataset(smiles_tokens=smiles_tokens[0:8284],
                              f=mogen_fp[0:8284],
                              f1=fp1[0:8284],
                              f2=fp2[0:8284],
                              f3=fp3[0:8284],
                              f4=fp4[0:8284],
                              X=X[0:8284],
                              A=A[0:8284],
                              label=labels[0:8284])
    valid_dataset = MyDataset(smiles_tokens=smiles_tokens[8284:],
                              f=mogen_fp[8284:],
                              f1=fp1[8284:],
                              f2=fp2[8284:],
                              f3=fp3[8284:],
                              f4=fp4[8284:],
                              X=X[8284:],
                              A=A[8284:],
                              label=labels[8284:])
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False, drop_last=True)

    model = MyModel(vocab_size=5000,
                    embed_dim=args.embed_dim,
                    num_heads=args.num_heads,
                    ff_dim=args.ff_dim,
                    num_layers=args.num_layers,
                    dropout=args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    num_epochs = 5
    log.write("Début de l'entraînement\n")
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            smiles_tokens_b, fp, fp1, fp2, fp3, fp4, X_batch, A_batch, label = batch
            optimizer.zero_grad()
            smiles_tokens_b = smiles_tokens_b.to(device)
            fp = fp.to(device)
            fp1 = fp1.to(device)
            fp2 = fp2.to(device)
            fp3 = fp3.to(device)
            fp4 = fp4.to(device)
            X_batch = X_batch.to(device)
            A_batch = A_batch.to(device)
            label = label.to(device)
            logits, loss = model(smiles_tokens_b, fp, fp1, fp2, fp3, fp4, X_batch, A_batch, label)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        log.write("Epoch {} - Loss: {:.4f}\n".format(epoch, loss.item()))
    # Évaluation sur validation : calcul de l'accuracy et de l'AUC
    model.eval()
    val_pred = []
    val_true = []
    with torch.no_grad():
        for batch in valid_loader:
            smiles_tokens_b, fp, fp1, fp2, fp3, fp4, X_batch, A_batch, label = batch
            smiles_tokens_b = smiles_tokens_b.to(device)
            fp = fp.to(device)
            fp1 = fp1.to(device)
            fp2 = fp2.to(device)
            fp3 = fp3.to(device)
            fp4 = fp4.to(device)
            X_batch = X_batch.to(device)
            A_batch = A_batch.to(device)
            label = label.to(device)
            logits, _ = model(smiles_tokens_b, fp, fp1, fp2, fp3, fp4, X_batch, A_batch, label)
            val_pred.extend(logits.detach().cpu().numpy())
            val_true.extend(label.detach().cpu().numpy())
    val_pred_bin = np.array(val_pred) > 0.5
    acc = accuracy_score(np.array(val_true), val_pred_bin)
    try:
        auc = roc_auc_score(np.array(val_true), np.array(val_pred))
    except Exception as e:
        auc = 0.0
    log.write("Validation Accuracy: {:.4f}\n".format(acc))
    log.write("Validation AUC: {:.4f}\n".format(auc))
    # Combinaison des métriques (ici la moyenne)
    combined_metric = (acc + auc) / 2.0
    return combined_metric, 0.0

##############################################
# 3. Hyperparameter Optimization avec Hyperopt
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
    args.embed_dim = 128
    args.num_heads = 8
    args.ff_dim = 256
    args.num_layers = 6
    args.dropout = 0.1
    args.lr = 1e-4
    args.save_path = "./checkpoints"
    args.log_path = "./logs"
    args.dataset_type = "classification"
    args.metric = "combined_metric"
    args.search_num = 3000
    args.search_now = 0
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    return args

space = {
    'embed_dim': hp.quniform('embed_dim', low=64, high=256, q=16),
    'num_heads': hp.quniform('num_heads', low=2, high=8, q=1),
    'ff_dim': hp.quniform('ff_dim', low=128, high=512, q=64),
    'num_layers': hp.quniform('num_layers', low=3, high=8, q=1),
    'dropout': hp.quniform('dropout', low=0.0, high=0.5, q=0.05),
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-3))
}

def objective(hyperparams):
    hyperparams['embed_dim'] = int(hyperparams['embed_dim'])
    hyperparams['num_heads'] = int(hyperparams['num_heads'])
    hyperparams['ff_dim'] = int(hyperparams['ff_dim'])
    hyperparams['num_layers'] = int(hyperparams['num_layers'])
    
    if hyperparams['embed_dim'] % hyperparams['num_heads'] != 0:
        print("Invalid hyperparameters: embed_dim {} not divisible by num_heads {}".format(
            hyperparams['embed_dim'], hyperparams['num_heads']))
        return 1e6
    
    args = set_hyper_argument()
    for key, value in hyperparams.items():
        setattr(args, key, value)
    
    dir_name = "embed{}_heads{}_ff{}_layers{}_drop{}_lr{}".format(
        hyperparams['embed_dim'],
        hyperparams['num_heads'],
        hyperparams['ff_dim'],
        hyperparams['num_layers'],
        hyperparams['dropout'],
        round(hyperparams['lr'], 6)
    )
    args.save_path = os.path.join(args.save_path, dir_name)
    log_name = "train_" + dir_name
    log = set_log(log_name, args.log_path)
    
    metric, std = training(args, log)
    
    result_path = os.path.join(args.log_path, "hyper_para_result.txt")
    with open(result_path, "a") as file:
        file.write("Hyperparams: " + str(hyperparams) + "\n")
        file.write("Result ({}): {:.4f} +/- {:.4f}\n".format(args.metric, metric, std))
    
    args.search_now += 1
    if args.dataset_type == "classification":
        return -metric
    else:
        return metric

def hyper_search(args):
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=args.search_num,
                trials=trials)
    result_path = os.path.join(args.log_path, "hyper_para_result.txt")
    with open(result_path, "a") as file:
        file.write("Best Hyperparameters:\n")
        file.write(str(best) + "\n")
    print("Meilleure configuration:", best)

##############################################
# 4. Exécution principale
##############################################
if __name__ == '__main__':
    args = set_hyper_argument()
    hyper_search(args)
