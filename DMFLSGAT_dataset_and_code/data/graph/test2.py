# eval_external_cpu.py

if __name__ == '__main__':
    import sys
    import pickle
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, confusion_matrix, roc_curve
    )

    # Ajouter ton dossier "data" au PYTHONPATH
    sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

    from utils import load_data_with_smiles, load_fp, MyDataset, set_seed
    from DMFLSGAT import MyModel

    ############################################################
    # 1. Initialisation
    ############################################################

    SEED = 42
    set_seed(SEED)

    # 🔥 CPU FORCÉ
    DEVICE = torch.device("cpu")
    print("Device utilisé :", DEVICE)

    MODEL_PATH = "/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/External dataset/tdmf_lsgat_internal.pth"
    SMILES_UTILS_PATH = "/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/External dataset/smiles_utils.pkl"

    ############################################################
    # 2. Charger smiles_utils + modèle entraîné
    ############################################################

    # 1) smiles_utils (tokenizer SMILES entraîné sur dataset interne)
    with open(SMILES_UTILS_PATH, "rb") as f:
        smiles_utils = pickle.load(f)
    print("✅ smiles_utils chargé")

    # 2) Modèle (chargé sur CPU)
    model = MyModel(vocab_size=5000)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()
    print("✅ Modèle chargé sur CPU")

    ############################################################
    # 3. Charger le DATASET EXTERNE
    ############################################################

    ext_smiles_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/External dataset/external_test_set_pos.csv'

    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/External dataset/external_test_set_pos_FPs/PubchemFP881.csv'
    Topological_torsion_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/External dataset/external_test_set_pos_FPs/TopologicalTorsion1024.csv'
    APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/External dataset/external_test_set_pos_FPs/APC2D780.csv'
    KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/External dataset/external_test_set_pos_FPs/KR.csv'

    
    # Charger graph + MorganFP + labels + smiles_tokens pour l'externe
    X_ext, A_ext, mogen_fp_ext, labels_ext, smiles_tokens_ext = load_data_with_smiles(
        ext_smiles_path,
        smiles_utils
    )

    # Charger les FPs externes
    fp1_ext = torch.FloatTensor(load_fp(PubchemFP881_path))
    fp2_ext = torch.FloatTensor(load_fp(Topological_torsion_path))
    fp3_ext = torch.FloatTensor(load_fp(APC2D780_path))
    fp4_ext = torch.FloatTensor(load_fp(KR_path))

    # Conversion en CPU tensors
    X_ext = torch.FloatTensor(X_ext)
    A_ext = torch.FloatTensor(A_ext)
    mogen_fp_ext = torch.FloatTensor(mogen_fp_ext)
    smiles_tokens_ext = torch.LongTensor(smiles_tokens_ext)
    labels_ext = torch.FloatTensor(labels_ext)

    # Dataset EXTERNE
    external_dataset = MyDataset(
        smiles_tokens=smiles_tokens_ext,
        f=mogen_fp_ext,
        f1=fp1_ext,
        f2=fp2_ext,
        f3=fp3_ext,
        f4=fp4_ext,
        X=X_ext,
        A=A_ext,
        label=labels_ext
    )

    external_loader = DataLoader(
        external_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False
    )

    ############################################################
    # 4. Évaluation sur le dataset EXTERNE (CPU)
    ############################################################

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in external_loader:
            smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch_data

            #  Tout sur CPU
            smiles_tokens_b = smiles_tokens_b.cpu()
            fp_b  = fp_b.cpu()
            fp1_b = fp1_b.cpu()
            fp2_b = fp2_b.cpu()
            fp3_b = fp3_b.cpu()
            fp4_b = fp4_b.cpu()
            X_b   = X_b.cpu()
            A_b   = A_b.cpu()
            label_b = label_b.cpu()

            logits, loss = model(
                smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b,
                X_b, A_b, label_b
            )

            probs = logits.detach().cpu().numpy().reshape(-1)
            labels_np = label_b.cpu().numpy().reshape(-1)

            all_probs.extend(probs)
            all_labels.extend(labels_np)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"\nNombre de molécules externes : {len(all_labels)}")
    print(f"Proportion hERG+ : {all_labels.mean():.3f}")
  

    # 2️⃣ Seuil via ROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Seuil optimal (ROC externe) : {optimal_threshold:.4f}")

    # 3️⃣ Binarisation
    y_pred_bin = (all_probs >= optimal_threshold).astype(int)

    cm = confusion_matrix(all_labels, y_pred_bin)

    try:
        TN, FP, FN, TP = cm.ravel()
    except:
        TN = FP = FN = TP = 0

    acc = accuracy_score(all_labels, y_pred_bin)
    auc_val = roc_auc_score(all_labels, all_probs)
    SPE = TN / (TN + FP) if (TN + FP) else 0
    SEN = TP / (TP + FN) if (TP + FN) else 0
    PPV = TP / (TP + FP) if (TP + FP) else 0
    NPV = TN / (TN + FN) if (TN + FN) else 0
    mcc_num = (TP * TN - FP * FN)
    mcc_den = max(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 1e-8)
    MCC = mcc_num / (mcc_den ** 0.5)

    print('\n========== Évaluation EXTERNE (CPU) ==========')
    print(f"External ACC : {acc:.3f}")
    print(f"External AUC : {auc_val:.3f}")
    print(f"Specificity (SPE): {SPE:.3f}")
    print(f"Sensitivity (SEN): {SEN:.3f}")
    print(f"NPV : {NPV:.3f}")
    print(f"PPV : {PPV:.3f}")
    print(f"MCC : {MCC:.3f}")


