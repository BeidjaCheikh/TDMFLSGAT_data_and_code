import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score
)

if __name__ == '__main__':
    from Utils import load_data, MyDataset, set_seed, load_fp
    from utils import DataLoader
    import pandas as pd

    SEED = 42
    set_seed(SEED)

    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/Smiles.csv'
    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/PubchemFP881.csv'
    GraphFP1024_path  = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/GraphFP1024.csv'
    APC2D780_path     = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/APC2D780.csv'
    FP1024_path       = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/FP1024.csv'

    X, A, mogen_fp, labels = load_data(path)

    fp1 = load_fp(PubchemFP881_path)
    fp2 = load_fp(GraphFP1024_path)
    fp3 = load_fp(APC2D780_path)
    fp4 = load_fp(FP1024_path)

    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    fp1 = torch.FloatTensor(fp1)
    fp2 = torch.FloatTensor(fp2)
    fp3 = torch.FloatTensor(fp3)
    fp4 = torch.FloatTensor(fp4)
    labels = torch.FloatTensor(labels)

    from sklearn.ensemble import RandomForestClassifier

    # Modèle Random Forest
    model = RandomForestClassifier(random_state=0)

    # Entraînement
    model.fit(mogen_fp[0:8000], labels[0:8000])

    # Prédictions
    pred = model.predict(mogen_fp[8000:10355])
    proba = model.predict_proba(mogen_fp[8000:10355])[:, 1]  # probabilités classe 1

    # Métriques
    TN, FP, FN, TP = confusion_matrix(labels[8000:10355], pred).ravel()
    SPE = TN / (TN + FP)
    SEN = TP / (TP + FN)
    NPV = TN / (TN + FN)
    PPV = TP / (TP + FP)

    MCC = (TP * TN - FP * FN) / (
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    ) ** 0.5

    acc = accuracy_score(labels[8000:10355], pred)
    auc = roc_auc_score(labels[8000:10355], pred)  # conservé tel quel
    AP  = average_precision_score(labels[8000:10355], proba)

    # Affichage (traduction + 3 décimales)
    print("=== RF ===")
    print("TN, FP, FN, TP:", TN, FP, FN, TP)
    print(
        "Specificity (SPE), Sensitivity (SEN), NPV, PPV, MCC:",
        f"{SPE:.3f}", f"{SEN:.3f}", f"{NPV:.3f}", f"{PPV:.3f}", f"{MCC:.3f}"
    )
    print("Test set accuracy (ACC):", f"{acc:.3f}")
    print("Test set AUC:", f"{auc:.3f}")
    print("AP (Average Precision / AUPRC):", f"{AP:.3f}")
