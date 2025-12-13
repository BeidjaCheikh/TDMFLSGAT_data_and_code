# main_cv.py

import sys
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

import argparse
import os
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    confusion_matrix
)

from graph.utils import load_data_with_smiles, load_fp
from utils import SmilesUtils, MyDataset, set_seed, get_logger, load_fold_data_tdmf
from DMFLSGAT import MyModel   # ton modèle global

# ---------------------------------------------------------
# 1. Entraînement sur 1 epoch
# ---------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for batch in train_loader:
        smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch
        smiles_tokens_b = smiles_tokens_b.to(device)
        fp_b  = fp_b.to(device)
        fp1_b = fp1_b.to(device)
        fp2_b = fp2_b.to(device)
        fp3_b = fp3_b.to(device)
        fp4_b = fp4_b.to(device)
        X_b   = X_b.to(device)
        A_b   = A_b.to(device)
        label_b = label_b.to(device)

        optimizer.zero_grad()
        probs, loss = model(smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b)
        loss.backward()
        optimizer.step()

        batch_size = label_b.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = (probs >= 0.5).float()
        correct += (preds == label_b).sum().item()

    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    return avg_loss, acc

# ---------------------------------------------------------
# 2. Évaluation + métriques complètes (TES FORMULES)
# ---------------------------------------------------------
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch
            smiles_tokens_b = smiles_tokens_b.to(device)
            fp_b  = fp_b.to(device)
            fp1_b = fp1_b.to(device)
            fp2_b = fp2_b.to(device)
            fp3_b = fp3_b.to(device)
            fp4_b = fp4_b.to(device)
            X_b   = X_b.to(device)
            A_b   = A_b.to(device)
            label_b = label_b.to(device)

            probs, loss = model(smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b)

            batch_size = label_b.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(label_b.detach().cpu().numpy().tolist())

    avg_loss = total_loss / total_samples

    pred_y = np.array(all_probs)           # probabilités
    true_y = np.array(all_labels).astype(int)

    # 1️⃣ Seuil 0.5 pour binariser
    PED = (pred_y >= 0.5).astype(int)

    # 2️⃣ Matrice de confusion
    cm = confusion_matrix(true_y, PED)
    try:
        TN, FP, FN, TP = cm.ravel()
    except Exception:
        TN = FP = FN = TP = 0

    # 3️⃣ Tes métriques EXACTES
    acc = accuracy_score(true_y, PED)
    try:
        auc_score = roc_auc_score(true_y, pred_y)
    except ValueError:
        auc_score = 0.0

    SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
    SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    mcc_num = (TP * TN - FP * FN)
    mcc_den = max(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 1e-8)
    MCC = mcc_num / (mcc_den ** 0.5)

    # Pour info : PRE / REC classiques (non moyennés sur les folds, juste log)
    try:
        PRE = precision_score(true_y, PED)
    except ValueError:
        PRE = 0.0
    try:
        REC = recall_score(true_y, PED)
    except ValueError:
        REC = 0.0

    return avg_loss, acc, PRE, REC, auc_score, SPE, SEN, NPV, PPV, MCC

# ---------------------------------------------------------
# 3. Main K-fold
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--train_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fold', type=int, default=5)  # k = 5
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='hERG')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # ------------ Chargement des données ------------
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

    full_dataset = MyDataset(
        smiles_tokens=smiles_tokens,
        f=mogen_fp,
        f1=fp1,
        f2=fp2,
        f3=fp3,
        f4=fp4,
        X=X,
        A=A,
        label=labels
    )

    # ------------ Logger ------------
    model_name = 'TDMFLSGAT'
    logf = f'log/ALL_clas_train_{args.dataset_name}_{model_name}.log'
    os.makedirs("log", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)
    logger = get_logger(logf)

    logger.info(f'Dataset: {args.dataset_name}  task: clas  train_epoch: {args.train_epoch}')

    # Listes pour moyenne sur les folds (validation)
    fold_result_auc = []
    fold_result_acc = []
    fold_result_spe = []
    fold_result_sen = []
    fold_result_ppv = []
    fold_result_npv = []
    fold_result_mcc = []

    # ------------ Boucle K-fold ------------
    for fol in range(args.fold):
        logger.info('==============================================================')
        logger.info(f'Fold {fol}')

        best_val_acc = 0.0
        best_test_acc = 0.0

        bs_best_auc = 0.0
        bs_best_acc = 0.0
        bs_best_spe = 0.0
        bs_best_sen = 0.0
        bs_best_ppv = 0.0
        bs_best_npv = 0.0
        bs_best_mcc = 0.0

        model = MyModel(vocab_size=5000).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

        train_loader, valid_loader, test_loader = load_fold_data_tdmf(
            fol, args.batch_size, 0, args.fold, full_dataset
        )

        for epoch in range(1, args.train_epoch + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)

            valid_loss, v_acc, v_pre, v_rec, v_auc, VSPE, VSEN, VNPV, VPPV, VMCC = eval_epoch(
                model, valid_loader, device
            )
            test_loss, t_acc, t_pre, t_rec, t_auc, TSPE, TSEN, TNPV, TPPV, TMCC = eval_epoch(
                model, test_loader, device
            )

            scheduler.step()

            logger.info(
                f'Fold {fol:02d} Epoch {epoch:03d} '
                f'TrainLoss: {train_loss:.4f} TrainAcc: {train_acc:.4f}'
            )
            logger.info(
                f'Fold {fol:02d} Epoch {epoch:03d} '
                f'ValidLoss: {valid_loss:.4f}  ValidAUC: {v_auc:.4f}  ValidAcc: {v_acc:.4f}  '
                f'ValidPRE: {v_pre:.4f} ValidREC: {v_rec:.4f}  '
                f'ValidSPE: {VSPE:.4f} ValidSEN: {VSEN:.4f} ValidNPV: {VNPV:.4f} '
                f'ValidPPV: {VPPV:.4f} ValidMCC: {VMCC:.4f}'
            )
            logger.info(
                f'Fold {fol:02d} Epoch {epoch:03d} '
                f'TestLoss: {test_loss:.4f}  TestAUC: {t_auc:.4f}  TestAcc: {t_acc:.4f}  '
                f'TestPRE: {t_pre:.4f} TestREC: {t_rec:.4f}  '
                f'TestSPE: {TSPE:.4f} TestSEN: {TSEN:.4f} TestNPV: {TNPV:.4f} '
                f'TestPPV: {TPPV:.4f} TestMCC: {TMCC:.4f}'
            )

            # --- Sauvegarde best valid (ACC) ---
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                bs_best_acc = v_acc
                bs_best_auc = v_auc
                bs_best_spe = VSPE
                bs_best_sen = VSEN
                bs_best_ppv = VPPV
                bs_best_npv = VNPV
                bs_best_mcc = VMCC

                torch.save(
                    model.state_dict(),
                    f'ckpt/fol{fol}_valid_{model_name}.pth'
                )
                print(f'[Fold {fol}] best valid ckpt saved (acc = {best_val_acc:.4f})')

            # --- Sauvegarde best test (ACC) ---
            if t_acc > best_test_acc:
                best_test_acc = t_acc
                torch.save(
                    model.state_dict(),
                    f'ckpt/fol{fol}_test_{model_name}.pth'
                )
                print(f'[Fold {fol}] best test ckpt saved (acc = {best_test_acc:.4f})')

        logger.info(
            f'Fold {fol} best_val_auc: {bs_best_auc:.4f} best_val_acc: {bs_best_acc:.4f} '
            f'best_val_SPE: {bs_best_spe:.4f} best_val_SEN: {bs_best_sen:.4f} '
            f'best_val_PPV: {bs_best_ppv:.4f} best_val_NPV: {bs_best_npv:.4f} '
            f'best_val_MCC: {bs_best_mcc:.4f}'
        )

        fold_result_auc.append(bs_best_auc)
        fold_result_acc.append(bs_best_acc)
        fold_result_spe.append(bs_best_spe)
        fold_result_sen.append(bs_best_sen)
        fold_result_ppv.append(bs_best_ppv)
        fold_result_npv.append(bs_best_npv)
        fold_result_mcc.append(bs_best_mcc)

    # ------------ Moyennes sur les folds (VALID) ------------
    ava_auc = sum(fold_result_auc) / len(fold_result_auc)
    ava_acc = sum(fold_result_acc) / len(fold_result_acc)
    ava_spe = sum(fold_result_spe) / len(fold_result_spe)
    ava_sen = sum(fold_result_sen) / len(fold_result_sen)
    ava_ppv = sum(fold_result_ppv) / len(fold_result_ppv)
    ava_npv = sum(fold_result_npv) / len(fold_result_npv)
    ava_mcc = sum(fold_result_mcc) / len(fold_result_mcc)

    logger.info('==============================================================')
    logger.info(
        f'Average over folds (VALID) -> '
        f'AUC: {ava_auc:.4f}  ACC: {ava_acc:.4f}  '
        f'SPE: {ava_spe:.4f}  SEN: {ava_sen:.4f}  '
        f'PPV: {ava_ppv:.4f}  NPV: {ava_npv:.4f}  '
        f'MCC: {ava_mcc:.4f}'
    )

if __name__ == '__main__':
    main()
