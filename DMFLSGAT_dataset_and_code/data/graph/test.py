# test.py

import sys
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)

# Adapter les chemins si besoin
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

from utils import SmilesUtils, MyDataset, load_data_with_smiles, load_fp, set_seed
from DMFLSGAT import MyModel    # ton modèle global

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    set_seed(42)

    ##########################################################
    # 1) Recharger exactement les mêmes données que train.py
    ##########################################################
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/Smiles.csv'
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()

    smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
    smiles_utils.train_tokenizer(smiles_list)

    # X_np : (N, maxAtoms, 65),  A_np : (N, maxAtoms, maxAtoms)
    # mogen_fp_np : (N, 1024),   labels_list : (N,), smiles_tokens_np : (N, max_len)
    X_np, A_np, mogen_fp_np, labels_list, smiles_tokens_np = load_data_with_smiles(path, smiles_utils)

    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/FP/PubchemFP881.csv'
    Topological_torsion_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/FP/topological_torsion_fingerprints.csv'
    APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/FP/APC2D780.csv'
    KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/FP/KR.csv'

    fp1_np = load_fp(PubchemFP881_path)
    fp2_np = load_fp(Topological_torsion_path)
    fp3_np = load_fp(APC2D780_path)
    fp4_np = load_fp(KR_path)

    # Conversion en tenseurs
    X = torch.FloatTensor(X_np)
    A = torch.FloatTensor(A_np)
    mogen_fp = torch.FloatTensor(mogen_fp_np)
    fp1 = torch.FloatTensor(fp1_np)
    fp2 = torch.FloatTensor(fp2_np)
    fp3 = torch.FloatTensor(fp3_np)
    fp4 = torch.FloatTensor(fp4_np)
    smiles_tokens = torch.LongTensor(smiles_tokens_np)
    labels = torch.FloatTensor(labels_list)

    ##########################################################
    # 2) Refaire le même split 80 % / 20 %
    ##########################################################
    N = len(labels)
    Count = int(0.8 * N)   # 80% pour train/val (k-fold), 20% pour test externe

    # 20% test = mêmes indices que dans train.py
    smiles_test = smiles_tokens[Count:]
    labels_test = labels[Count:]
    f_test  = mogen_fp[Count:]
    f1_test = fp1[Count:]
    f2_test = fp2[Count:]
    f3_test = fp3[Count:]
    f4_test = fp4[Count:]
    X_test  = X[Count:]
    A_test  = A[Count:]

    ##########################################################
    # 3) Construire le DataLoader de test
    ##########################################################
    test_dataset = MyDataset(
        smiles_tokens=smiles_test,
        f=f_test,
        f1=f1_test,
        f2=f2_test,
        f3=f3_test,
        f4=f4_test,
        X=X_test,
        A=A_test,
        label=labels_test
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False
    )

    ##########################################################
    # 4) Charger le modèle et les poids checkpoint.pt
    ##########################################################
    vocab_size = 5000
    model = MyModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load('/home/enset/Téléchargements/DMFGAM_data_and_code12/checkpoint.pt', map_location=device))
    model.to(device)
    model.eval()

    ##########################################################
    # 5) Boucle d'évaluation sur le test set
    ##########################################################
    all_true = []
    all_probs = []

    TP = TN = FP = FN = 0

    with torch.no_grad():
        for batch_data in test_loader:
            smiles_b, f_b, f1_b, f2_b, f3_b, f4_b, X_b, A_b, label_b = batch_data

            smiles_b = smiles_b.to(device)
            f_b  = f_b.to(device)
            f1_b = f1_b.to(device)
            f2_b = f2_b.to(device)
            f3_b = f3_b.to(device)
            f4_b = f4_b.to(device)
            X_b  = X_b.to(device)
            A_b  = A_b.to(device)
            label_b = label_b.to(device)

            # forward
            preds, loss = model(smiles_b, f_b, f1_b, f2_b, f3_b, f4_b, X_b, A_b, label_b)
            # preds : proba (batch,) entre 0 et 1

            probs_np = preds.detach().cpu().numpy()
            true_np  = label_b.detach().cpu().numpy()

            all_probs.extend(probs_np)
            all_true.extend(true_np)

    all_true = np.array(all_true).astype(int)   # labels 0/1
    all_probs = np.array(all_probs)

    ##########################################################
    # 6) Calcul des métriques (comme dans ton main.py)
    ##########################################################

    # Seuil : ici 0.5 (tu peux remplacer par un seuil optimal ROC si tu veux)
    PED = (all_probs >= 0.5).astype(int)

    # Matrice de confusion
    cm = confusion_matrix(all_true, PED)
    try:
        TN, FP, FN, TP = cm.ravel()
    except Exception as e:
        print("Erreur dans la matrice de confusion (classes déséquilibrées ?) :", e)
        TN = FP = FN = TP = 0

    acc = accuracy_score(all_true, PED)
    try:
        auc_score = roc_auc_score(all_true, all_probs)
    except Exception:
        auc_score = 0.0

    SPE = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    SEN = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0.0

    mcc_num = (TP * TN - FP * FN)
    mcc_den = max(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 1e-8)
    MCC = mcc_num / (mcc_den ** 0.5)

    ##########################################################
    # 7) Affichage propre des résultats sur le test set
    ##########################################################
    print("\n========== Test externe (20% des données) ==========")
    print(f"Test Accuracy (ACC): {acc:.3f}")
    print(f"Test AUC: {auc_score:.3f}")
    print(f"Specificity (SPE): {SPE:.3f}")
    print(f"Sensitivity (SEN): {SEN:.3f}")
    print(f"Positive Predictive Value (PPV): {PPV:.3f}")
    print(f"Negative Predictive Value (NPV): {NPV:.3f}")
    print(f"Matthews Correlation Coefficient (MCC): {MCC:.3f}")
    print("Confusion matrix [ [TN FP] [FN TP] ] =")
    print(cm)
