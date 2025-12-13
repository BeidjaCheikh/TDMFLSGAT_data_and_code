# train.py

import sys
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

# Adapter le chemin à ton projet si besoin
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

# Selon ton organisation de fichiers :
from graph.utils import load_data_with_smiles, set_seed, load_fp
from utils import SmilesUtils, MyDataset
from DMFLSGAT import MyModel           # ton modèle global
from DMFLSGAT_dataset_and_code.data.graph.earlystopping import EarlyStopping  # ton EarlyStopping existant

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##########################################################
#  A. get_kfold_data : même logique que le code DGL      #
##########################################################
def get_kfold_data(
    k, i,
    smiles_tokens, labels,
    f, f1, f2, f3, f4,
    X, A
):
    """
    Découpe les données (pour la partie train, pas le test final)
    en train / validation pour le fold i.

    Retourne :
      (smiles_train, labels_train, f_train, f1_train, f2_train, f3_train, f4_train, X_train, A_train,
       smiles_val,   labels_val,   f_val,   f1_val,   f2_val,   f3_val,   f4_val,   X_val,   A_val)
    """
    n = len(labels)
    fold_size = n // k
    val_start = i * fold_size

    if i != k - 1:
        val_end = (i + 1) * fold_size

        # Validation : [val_start : val_end]
        smiles_val = smiles_tokens[val_start:val_end]
        labels_val = labels[val_start:val_end]
        f_val  = f[val_start:val_end]
        f1_val = f1[val_start:val_end]
        f2_val = f2[val_start:val_end]
        f3_val = f3[val_start:val_end]
        f4_val = f4[val_start:val_end]
        X_val  = X[val_start:val_end]
        A_val  = A[val_start:val_end]

        # Train : [0:val_start] U [val_end:n]
        smiles_train = torch.cat((smiles_tokens[0:val_start],
                                  smiles_tokens[val_end:]), dim=0)
        labels_train = torch.cat((labels[0:val_start],
                                  labels[val_end:]), dim=0)
        f_train  = torch.cat((f[0:val_start],  f[val_end:]),  dim=0)
        f1_train = torch.cat((f1[0:val_start], f1[val_end:]), dim=0)
        f2_train = torch.cat((f2[0:val_start], f2[val_end:]), dim=0)
        f3_train = torch.cat((f3[0:val_start], f3[val_end:]), dim=0)
        f4_train = torch.cat((f4[0:val_start], f4[val_end:]), dim=0)
        X_train  = torch.cat((X[0:val_start],  X[val_end:]),  dim=0)
        A_train  = torch.cat((A[0:val_start],  A[val_end:]),  dim=0)
    else:
        # Dernier fold
        smiles_val = smiles_tokens[val_start:]
        labels_val = labels[val_start:]
        f_val  = f[val_start:]
        f1_val = f1[val_start:]
        f2_val = f2[val_start:]
        f3_val = f3[val_start:]
        f4_val = f4[val_start:]
        X_val  = X[val_start:]
        A_val  = A[val_start:]

        smiles_train = smiles_tokens[0:val_start]
        labels_train = labels[0:val_start]
        f_train  = f[0:val_start]
        f1_train = f1[0:val_start]
        f2_train = f2[0:val_start]
        f3_train = f3[0:val_start]
        f4_train = f4[0:val_start]
        X_train  = X[0:val_start]
        A_train  = A[0:val_start]

    return (
        smiles_train, labels_train,
        f_train, f1_train, f2_train, f3_train, f4_train,
        X_train, A_train,
        smiles_val, labels_val,
        f_val, f1_val, f2_val, f3_val, f4_val,
        X_val, A_val
    )


##########################################################
#  B. traink : un fold avec toutes les métriques         #
##########################################################
def traink(
    model,
    smiles_train, labels_train,
    f_train, f1_train, f2_train, f3_train, f4_train,
    X_train, A_train,
    smiles_val, labels_val,
    f_val, f1_val, f2_val, f3_val, f4_val,
    X_val, A_val,
    BATCH_SIZE, learning_rate, TOTAL_EPOCHS, fold_id=1
):

    # Dataset train
    train_dataset = MyDataset(
        smiles_tokens=smiles_train,
        f=f_train, f1=f1_train, f2=f2_train, f3=f3_train, f4=f4_train,
        X=X_train, A=A_train,
        label=labels_train
    )
    # Dataset val
    val_dataset = MyDataset(
        smiles_tokens=smiles_val,
        f=f_val, f1=f1_val, f2=f2_val, f3=f3_val, f4=f4_val,
        X=X_val, A=A_val,
        label=labels_val
    )

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   BATCH_SIZE, shuffle=True)

    # IMPORTANT : on utilise le modèle passé en argument
    model = model.to(device)

    # Optimiseur + CosineAnnealingLR
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_EPOCHS,   # nombre total d’epochs prévu
        eta_min=0.0
    )

    # EarlyStopping basé sur l’AUC
    early_stopping = EarlyStopping(patience=10, verbose=True)

    losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    AUC_list = []
    SEN_list = []
    SPE_list = []
    ACC_list = []
    PPV_list = []
    NPV_list = []
    MCC_list = []

    n_train = len(labels_train)
    n_val   = len(labels_val)

    for epoch in range(TOTAL_EPOCHS):
        ########################
        #        TRAIN         #
        ########################
        model.train()
        correct_train = 0
        epoch_train_losses = []

        for batch_data in train_loader:
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

            optimizer.zero_grad()
            preds, loss = model(
                smiles_b, f_b, f1_b, f2_b, f3_b, f4_b, X_b, A_b, label_b
            )
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

            pred_labels = (preds >= 0.5).float()
            correct_train += (pred_labels == label_b).sum().item()

        mean_train_loss = float(np.mean(epoch_train_losses))
        train_accuracy = 100.0 * correct_train / n_train

        losses.append(mean_train_loss)
        train_acc.append(train_accuracy)

        print(f'[Fold {fold_id}] Epoch: {epoch+1}/{TOTAL_EPOCHS}, '
              f'Train loss: {mean_train_loss:.5f}, '
              f'Train acc: {correct_train}/{n_train} ({train_accuracy:.3f}%)')

        ########################
        #      VALIDATION      #
        ########################
        model.eval()
        val_loss_sum = 0.0
        correct_val = 0
        all_true = []
        all_probs = []

        with torch.no_grad():
            for batch_data in val_loader:
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

                preds, loss = model(
                    smiles_b, f_b, f1_b, f2_b, f3_b, f4_b, X_b, A_b, label_b
                )
                bs = label_b.size(0)
                val_loss_sum += loss.item() * bs

                all_true.extend(label_b.cpu().numpy())
                all_probs.extend(preds.cpu().numpy())

                pred_labels = (preds >= 0.5).float()
                correct_val += (pred_labels == label_b).sum().item()

        mean_val_loss = val_loss_sum / n_val
        val_accuracy = 100.0 * correct_val / n_val
        val_losses.append(mean_val_loss)
        val_acc.append(val_accuracy)

        # Métriques détaillées sur la validation
        all_true_np = np.array(all_true)
        all_probs_np = np.array(all_probs)

        PED = (all_probs_np >= 0.5).astype(int)

        try:
            auc_score = roc_auc_score(all_true_np, all_probs_np)
        except Exception:
            auc_score = 0.0

        try:
            TN, FP, FN, TP = confusion_matrix(all_true_np, PED).ravel()
        except ValueError:
            TN = FP = FN = TP = 0

        acc = (TP + TN) / max((TP + TN + FP + FN), 1e-8)
        SPE = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        SEN = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        NPV = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        mcc_num = (TP * TN - FP * FN)
        mcc_den = max(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 1e-8)
        MCC = mcc_num / (mcc_den ** 0.5)

        AUC_list.append(auc_score)
        SEN_list.append(SEN)
        SPE_list.append(SPE)
        ACC_list.append(acc)
        PPV_list.append(PPV)
        NPV_list.append(NPV)
        MCC_list.append(MCC)

        print(
            '[Fold {}] Val: loss: {:.4f}, acc: {}/{} ({:.3f}%) | '
            'AUC: {:.3f}, ACC: {:.3f}, SEN: {:.3f}, SPE: {:.3f}, '
            'PPV: {:.3f}, NPV: {:.3f}, MCC: {:.3f}\n'.format(
                fold_id, mean_val_loss, correct_val, n_val, val_accuracy,
                auc_score, acc, SEN, SPE, PPV, NPV, MCC
            )
        )

        # 🔁 CosineAnnealingLR : on step à CHAQUE epoch, sans argument
        lr_scheduler.step()

        # EarlyStopping sur l’AUC
        early_stopping(auc_score, model)
        if early_stopping.early_stop:
            print(f"Early stopping on fold {fold_id}")
            break

    return (
        losses, val_losses, train_acc, val_acc,
        AUC_list, SEN_list, SPE_list, ACC_list, PPV_list, NPV_list, MCC_list
    )


##########################################################
#  C. k_fold : agrège les résultats des k folds          #
##########################################################
def k_fold(
    k,
    smiles_tokens, labels,
    f, f1, f2, f3, f4,
    X, A,
    num_epochs=150, learning_rate=1e-4, batch_size=100, vocab_size=5000
):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0

    AUC_sum = 0
    SEN_sum = 0
    SPE_sum = 0
    ACC_sum = 0
    PPV_sum = 0
    NPV_sum = 0
    MCC_sum = 0

    for i in range(k):
        print('*' * 25, f'Fold {i+1}', '*' * 25)
        data = get_kfold_data(
            k, i,
            smiles_tokens, labels,
            f, f1, f2, f3, f4,
            X, A
        )

        model = MyModel(vocab_size=vocab_size)

        (train_loss, val_loss, train_acc, val_acc,
         AUC_list, SEN_list, SPE_list, ACC_list, PPV_list, NPV_list, MCC_list) = traink(
            model,
            *data,
            BATCH_SIZE=batch_size,
            learning_rate=learning_rate,
            TOTAL_EPOCHS=num_epochs,
            fold_id=i+1
        )

        # Meilleure epoch = max AUC sur ce fold
        best_idx = int(np.argmax(AUC_list))

        print('Best epoch on fold {}:'.format(i+1))
        print('  train_loss:{:.5f}, train_acc:{:.3f}%'.format(
            train_loss[best_idx], train_acc[best_idx]
        ))
        print('  valid loss:{:.5f}, valid_acc:{:.3f}%'.format(
            val_loss[best_idx], val_acc[best_idx]
        ))
        print('  AUC:{:.3f}, ACC:{:.3f}, SEN:{:.3f}, SPE:{:.3f}, '
              'PPV:{:.3f}, NPV:{:.3f}, MCC:{:.3f}\n'.format(
                  AUC_list[best_idx], ACC_list[best_idx],
                  SEN_list[best_idx], SPE_list[best_idx],
                  PPV_list[best_idx], NPV_list[best_idx],
                  MCC_list[best_idx]
              ))

        train_loss_sum += train_loss[best_idx]
        valid_loss_sum += val_loss[best_idx]
        train_acc_sum  += train_acc[best_idx]
        valid_acc_sum  += val_acc[best_idx]

        AUC_sum += AUC_list[best_idx]
        SEN_sum += SEN_list[best_idx]
        SPE_sum += SPE_list[best_idx]
        ACC_sum += ACC_list[best_idx]
        PPV_sum += PPV_list[best_idx]
        NPV_sum += NPV_list[best_idx]
        MCC_sum += MCC_list[best_idx]

    print('\n', '#' * 10, 'Résultats finaux k-fold', '#' * 10)
    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'
          .format(train_loss_sum / k, train_acc_sum / k))
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'
          .format(valid_loss_sum / k, valid_acc_sum / k))
    print('average valid AUC:{:.3f}, average valid ACC:{:.3f}, '
          'average SEN:{:.3f}, average SPE:{:.3f}'.format(
              AUC_sum / k, ACC_sum / k, SEN_sum / k, SPE_sum / k
          ))
    print('average PPV:{:.3f}, average NPV:{:.3f}, average MCC:{:.3f}'
          .format(PPV_sum / k, NPV_sum / k, MCC_sum / k))


##########################################################
#  D. main : 80% train/val (k-fold) + 20% test externe   #
##########################################################
if __name__ == '__main__':
    set_seed(42)

    # 1) Charger les données complètes
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/Smiles.csv'
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()

    smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
    smiles_utils.train_tokenizer(smiles_list)

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

    N = len(labels)
    Count = int(0.8 * N)   # 80% pour train/val (k-fold), 20% pour test externe

    # 80% pour k-fold
    smiles_train_all = smiles_tokens[:Count]
    labels_train_all = labels[:Count]
    f_train_all  = mogen_fp[:Count]
    f1_train_all = fp1[:Count]
    f2_train_all = fp2[:Count]
    f3_train_all = fp3[:Count]
    f4_train_all = fp4[:Count]
    X_train_all  = X[:Count]
    A_train_all  = A[:Count]


    # 2) Lancer le k-fold sur les 80% train/val
    k_fold(
        k=5,
        smiles_tokens=smiles_train_all,
        labels=labels_train_all,
        f=f_train_all,
        f1=f1_train_all,
        f2=f2_train_all,
        f3=f3_train_all,
        f4=f4_train_all,
        X=X_train_all,
        A=A_train_all,
        num_epochs=150,
        learning_rate=1e-4,
        batch_size=250,
        vocab_size=5000
    )
