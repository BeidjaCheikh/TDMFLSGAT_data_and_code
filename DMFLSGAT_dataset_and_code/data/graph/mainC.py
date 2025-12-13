# mainC.py  (ou main_beta_sweep.py)

import sys
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from graph.utils import load_data_with_smiles, set_seed, load_fp
from utils import SmilesUtils, MyDataset
from DMFGAM数据集及代码.data.graph.DMFGAT import MyModel

from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve
)

#############################################
# 0. Constante de seed
#############################################
SEED = 42

#############################################
# 1. Chargement des données (comme main.py)
#############################################
path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/Smiles.csv'
data = pd.read_csv(path)
smiles_list = data['smiles'].tolist()

smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
smiles_utils.train_tokenizer(smiles_list)

X, A, mogen_fp, labels, smiles_tokens = load_data_with_smiles(path, smiles_utils)

PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/FP/PubchemFP881.csv'
Topological_torsion_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/FP/topological_torsion_fingerprints.csv'
APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/FP/APC2D780.csv'
KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/FP/KR.csv'

fp1 = torch.FloatTensor(load_fp(PubchemFP881_path))
fp2 = torch.FloatTensor(load_fp(Topological_torsion_path))
fp3 = torch.FloatTensor(load_fp(APC2D780_path))
fp4 = torch.FloatTensor(load_fp(KR_path))

X = torch.FloatTensor(X)
A = torch.FloatTensor(A)
mogen_fp = torch.FloatTensor(mogen_fp)
smiles_tokens = torch.LongTensor(smiles_tokens)
labels = torch.FloatTensor(labels)

# Split train/test identique à ton main.py
split_idx = 8284

train_dataset = MyDataset(
    smiles_tokens=smiles_tokens[:split_idx],
    f=mogen_fp[:split_idx],
    f1=fp1[:split_idx],
    f2=fp2[:split_idx],
    f3=fp3[:split_idx],
    f4=fp4[:split_idx],
    X=X[:split_idx],
    A=A[:split_idx],
    label=labels[:split_idx]
)
test_dataset = MyDataset(
    smiles_tokens=smiles_tokens[split_idx:],
    f=mogen_fp[split_idx:],
    f1=fp1[split_idx:],
    f2=fp2[split_idx:],
    f3=fp3[split_idx:],
    f4=fp4[split_idx:],
    X=X[split_idx:],
    A=A[split_idx:],
    label=labels[split_idx:]
)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=True)

#############################################
# 2. Sweep sur différentes valeurs de β
#############################################
beta_values = [0.2, 0.4, 0.6, 0.8, 1.0]

# On stocke : beta, seuil, ACC, AUC, SEN, SPE, PPV, NPV, MCC
results = []

for beta in beta_values:
    print(f"\n========== Training with beta = {beta} ==========")

    # 🔁 Comme si tu relançais le script à chaque fois :
    # même seed, même point de départ, seule différence = beta
    set_seed(SEED)

    # Modèle avec ce beta
    model = MyModel(vocab_size=5000, beta_lsgat=beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    num_epochs = 30
    model.train()
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch_data

            optimizer.zero_grad()
            logits, loss = model(smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    #############################################
    # 3. Évaluation sur le test set
    #############################################
    model.eval()
    pred_y, true_y = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch_data
            logits, _ = model(smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b)
            logits_np = logits.cpu().numpy()
            pred_y.extend(logits_np)
            true_y.extend(label_b.cpu().numpy())

    pred_y = np.array(pred_y)
    true_y = np.array(true_y)

    # ✅ Seuil optimal via ROC (comme dans ton main.py)
    fpr, tpr, thresholds = roc_curve(true_y, pred_y)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"[beta={beta}] Seuil optimal ROC : {optimal_threshold:.4f}")

    # Binarisation avec ce seuil
    PED = (pred_y >= optimal_threshold).astype(int)

    cm = confusion_matrix(true_y, PED)
    try:
        TN, FP, FN, TP = cm.ravel()
    except Exception as e:
        print("Problème matrice de confusion, classes déséquilibrées ?", e)
        TN = FP = FN = TP = 0

    acc = accuracy_score(true_y, PED)
    auc_score = roc_auc_score(true_y, pred_y)
    SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
    SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    mcc_num = (TP * TN - FP * FN)
    mcc_den = max(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 1e-8)
    MCC = mcc_num / (mcc_den ** 0.5)

    print(
        f"[beta={beta}] ACC={acc:.3f}, AUC={auc_score:.3f}, "
        f"SEN={SEN:.3f}, SPE={SPE:.3f}, PPV={PPV:.3f}, NPV={NPV:.3f}, MCC={MCC:.3f}"
    )

    # On stocke tout
    results.append((beta, optimal_threshold, acc, auc_score, SEN, SPE, PPV, NPV, MCC))

#############################################
# 4. Sauvegarde CSV + TXT
#############################################
df_results = pd.DataFrame(
    results,
    columns=["beta", "threshold", "ACC", "AUC", "SEN", "SPE", "PPV", "NPV", "MCC"]
)

df_results.to_csv("beta_sweep_results.csv", index=False)
print("\nFichier CSV enregistré sous : beta_sweep_results.csv")

with open("beta_sweep_results.txt", "w") as f:
    f.write("Sensitivity Analysis of LSGAT beta\n")
    f.write("=================================\n\n")
    for beta, thr, acc, auc_score, SEN, SPE, PPV, NPV, MCC in results:
        f.write(
            f"beta={beta:.1f} | thr={thr:.4f} | "
            f"ACC={acc:.3f} | AUC={auc_score:.3f} | "
            f"SEN={SEN:.3f} | SPE={SPE:.3f} | PPV={PPV:.3f} | "
            f"NPV={NPV:.3f} | MCC={MCC:.3f}\n"
        )

print("Fichier TXT enregistré sous : beta_sweep_results.txt")

print("\n===== Résumé final =====")
for beta, thr, acc, auc_score, SEN, SPE, PPV, NPV, MCC in results:
    print(
        f"beta={beta:.1f} | thr={thr:.4f} | "
        f"ACC={acc:.3f} | AUC={auc_score:.3f} | "
        f"SEN={SEN:.3f} | SPE={SPE:.3f} | PPV={PPV:.3f} | "
        f"NPV={NPV:.3f} | MCC={MCC:.3f}"
    )
