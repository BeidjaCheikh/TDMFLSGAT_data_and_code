# train_internal.py

if __name__ == '__main__':
    import sys
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, confusion_matrix, roc_curve
    )

    sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

    from utils import load_data_with_smiles, set_seed, load_fp, SmilesUtils, MyDataset
    from DMFLSGAT_VZ import MyModel

    ############################################################
    #                  A. Chargement des données                #
    ############################################################
    SEED = 42
    set_seed(SEED)

    DEVICE = torch.device("cpu")  # CPU pour être sûr

    # 1) Chargement du CSV SMILES
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/Smiles.csv'
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()

    smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
    smiles_utils.train_tokenizer(smiles_list)

    X, A, mogen_fp, labels, smiles_tokens = load_data_with_smiles(path, smiles_utils)

    # 2) Chargement des autres FPs
    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/PubchemFP881.csv'
    Topological_torsion_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/topological_torsion_fingerprints.csv'
    APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/APC2D780.csv'
    KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/KR.csv'

    fp1 = torch.FloatTensor(load_fp(PubchemFP881_path))
    fp2 = torch.FloatTensor(load_fp(Topological_torsion_path))
    fp3 = torch.FloatTensor(load_fp(APC2D780_path))
    fp4 = torch.FloatTensor(load_fp(KR_path))

    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    smiles_tokens = torch.LongTensor(smiles_tokens)
    labels = torch.FloatTensor(labels)

    # Split train/test
    split_idx = 8284  # à adapter si besoin
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
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)

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
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=False)

    ############################################################
    #                B. Définition du modèle                   #
    ############################################################
    model = MyModel(vocab_size=5000).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

    ############################################################
    #                C. Entraînement du modèle                 #
    ############################################################
    model.train()
    num_epochs = 5
    all_epoch_losses = []

    for epoch in range(num_epochs):
        pred_y, PED, true_y = [], [], []
        epoch_losses = []

        for i, batch_data in enumerate(train_loader):
            smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch_data
            smiles_tokens_b = smiles_tokens_b.to(DEVICE)
            fp_b = fp_b.to(DEVICE)
            fp1_b = fp1_b.to(DEVICE)
            fp2_b = fp2_b.to(DEVICE)
            fp3_b = fp3_b.to(DEVICE)
            fp4_b = fp4_b.to(DEVICE)
            X_b = X_b.to(DEVICE)
            A_b = A_b.to(DEVICE)
            label_b = label_b.to(DEVICE)

            optimizer.zero_grad()
            logits, loss = model(smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            batch_loss_value = loss.item()
            epoch_losses.append(batch_loss_value)
            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}, Loss: {batch_loss_value:.4f}")

            logits_np = logits.detach().cpu().numpy()
            pred_y.extend(logits_np)
            PED.extend(np.round(logits_np))
            true_y.extend(label_b.cpu().numpy())

        mean_epoch_loss = np.mean(epoch_losses)
        all_epoch_losses.append(mean_epoch_loss)
        print(f"===> Epoch {epoch+1} finished. Mean Loss: {mean_epoch_loss:.4f}\n")

        if epoch == (num_epochs - 1):
            acc = accuracy_score(true_y, PED)
            auc_score = roc_auc_score(true_y, pred_y)
            print('[TRAIN] Accuracy:', round(acc, 3))
            print('[TRAIN] AUC:', round(auc_score, 3))

    # Courbe de loss
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, num_epochs+1), all_epoch_losses, marker='o', label='Mean loss per epoch')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_tdmflsgat.png", dpi=300)
    plt.show()

    ############################################################
    #               D. Évaluation sur Test Set                 #
    ############################################################
    model.eval()
    pred_y, true_y = [], []
    all_labels, all_probs = [], []

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch_data
            smiles_tokens_b = smiles_tokens_b.to(DEVICE)
            fp_b = fp_b.to(DEVICE)
            fp1_b = fp1_b.to(DEVICE)
            fp2_b = fp2_b.to(DEVICE)
            fp3_b = fp3_b.to(DEVICE)
            fp4_b = fp4_b.to(DEVICE)
            X_b = X_b.to(DEVICE)
            A_b = A_b.to(DEVICE)
            label_b = label_b.to(DEVICE)

            logits, _ = model(smiles_tokens_b, fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b)
            logits_np = logits.detach().cpu().numpy()

            pred_y.extend(logits_np)
            true_y.extend(label_b.cpu().numpy())
            all_labels.extend(label_b.cpu().numpy())
            all_probs.extend(logits_np)

    # Seuil optimal via ROC
    fpr, tpr, thresholds = roc_curve(true_y, pred_y)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Seuil optimal via ROC : {optimal_threshold:.4f}")

    PED = (np.array(pred_y) >= optimal_threshold).astype(int)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_y, PED)
    try:
        TN, FP, FN, TP = cm.ravel()
    except Exception as e:
        print("Erreur dans la matrice de confusion:", e)
        TN = FP = FN = TP = 0

    acc = accuracy_score(true_y, PED)
    auc_score = roc_auc_score(true_y, pred_y)
    SPE = TN / (TN + FP) if (TN + FP) > 0 else 0
    SEN = TP / (TP + FN) if (TP + FN) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    mcc_num = (TP * TN - FP * FN)
    mcc_den = max(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 1e-8)
    MCC = mcc_num / (mcc_den ** 0.5)

    print('\n========== Test Set ==========')
    print(f"ACC : {acc:.3f}")
    print(f"AUC : {auc_score:.3f}")
    print(f"SPE : {SPE:.3f}")
    print(f"SEN : {SEN:.3f}")
    print(f"NPV : {NPV:.3f}")
    print(f"PPV : {PPV:.3f}")
    print(f"MCC : {MCC:.3f}")

    # Matrice de confusion
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Pred: 0", "Pred: 1"],
                yticklabels=["True: 0", "True: 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_tdmflsgat.png", dpi=300)
    plt.show()

    # ROC curve
    from sklearn.metrics import auc
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], '--', lw=1, label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve_tdmflsgat.png", dpi=300)
    plt.show()

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "tdmflsgat_internal_best.pth")
    print("✅ Modèle sauvegardé sous : tdmflsgat_internal_best.pth")
