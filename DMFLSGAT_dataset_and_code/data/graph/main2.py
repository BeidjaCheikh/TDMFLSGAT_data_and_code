if __name__ == '__main__':
    import sys
    sys.path.append('/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from graph.utils2 import load_data_with_smiles, set_seed, load_fp
    from torch.utils.data import DataLoader
    from utils2 import SmilesUtils, MyDataset
    import pandas as pd
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
    import numpy as np
    from model2 import MyModel

    SEED = 42
    set_seed(SEED)

    # Chargement CSV SMILES
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/Smiles.csv'

    X, A, mogen_fp, labels = load_data_with_smiles(path)

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

    labels = torch.FloatTensor(labels)

    # Split train/test
    split_idx = 8284  # À ajuster si besoin
    train_dataset = MyDataset(
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
        f=mogen_fp[split_idx:],
        f1=fp1[split_idx:],
        f2=fp2[split_idx:],
        f3=fp3[split_idx:],
        f4=fp4[split_idx:],
        X=X[split_idx:],
        A=A[split_idx:],
        label=labels[split_idx:]
    )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=True)

    ############################################################
    #                B. Définition du modèle                   #
    ############################################################

    model = MyModel(vocab_size=5000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

    ############################################################
    #                C. Entraînement du modèle                 #
    ############################################################

    model.train()
    num_epochs = 5
    all_epoch_losses = []  # Pour stocker la loss moyenne par epoch

    for epoch in range(num_epochs):
        pred_y, PED, true_y = [], [], []
        epoch_losses = []  # Stocke toutes les losses de l’epoch

        for i, batch_data in enumerate(train_loader):
            fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch_data
            optimizer.zero_grad()

            logits, loss = model( fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Stockage et affichage de la loss du batch
            batch_loss_value = loss.item()
            epoch_losses.append(batch_loss_value)
            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}, Loss: {batch_loss_value:.4f}")

            logits_np = logits.detach().cpu().numpy()
            pred_y.extend(logits_np)
            PED.extend(np.round(logits_np))
            true_y.extend(label_b.cpu().numpy())

        # Calcul et affichage de la loss moyenne de l’epoch
        mean_epoch_loss = np.mean(epoch_losses)
        all_epoch_losses.append(mean_epoch_loss)
        print(f"===> Epoch {epoch+1} finished. Mean Loss: {mean_epoch_loss:.4f}\n")

        if epoch == (num_epochs - 1):
            acc = accuracy_score(true_y, PED)
            auc_score = roc_auc_score(true_y, pred_y)
            print('[TRAIN] Accuracy:', round(acc, 3))
            print('[TRAIN] AUC:', round(auc_score, 3))

    # Affichage de la courbe loss
    plt.figure(figsize=(7,5))
    plt.plot(range(1, num_epochs+1), all_epoch_losses, marker='o', label='Loss moyenne par epoch')
    plt.title('Courbe de la Loss durant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    ############################################################
    #               D. Évaluation et Matrice de confusion       #
    ############################################################
    model.eval()
    pred_y, true_y = [], []
    all_labels, all_probs = [], []

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b = batch_data
            logits, loss = model( fp_b, fp1_b, fp2_b, fp3_b, fp4_b, X_b, A_b, label_b)
            logits_np = logits.detach().cpu().numpy()
            pred_y.extend(logits_np)  # Probabilités
            true_y.extend(label_b.cpu().numpy())
            all_labels.extend(label_b.cpu().numpy())
            all_probs.extend(logits_np)

    # 1️⃣ Visualiser la distribution des probabilités
    plt.figure(figsize=(6, 4))
    plt.hist(pred_y, bins=50, color='blue', edgecolor='black')
    plt.title('Distribution des probabilités prédites')
    plt.xlabel('Probabilité')
    plt.ylabel('Nombre')
    plt.grid(True)
    plt.show()

    # 2️⃣ Trouver le seuil optimal basé sur ROC
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(true_y, pred_y)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Seuil optimal trouvé via ROC : {optimal_threshold:.4f}")

    # 3️⃣ Appliquer le seuil optimal pour binariser les probabilités
    PED = (np.array(pred_y) >= optimal_threshold).astype(int)

    # 4️⃣ Calcul des métriques finales
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

    cm = confusion_matrix(true_y, PED)
    try:
        TN, FP, FN, TP = cm.ravel()
    except Exception as e:
        print("Erreur dans la matrice de confusion (classes déséquilibrées ?):", e)
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

    # 5️⃣ Affichage des résultats
    print('\n========== Évaluation sur Test Set ==========')
    print(f"Test Accuracy (ACC): {acc:.3f}")
    print(f"Test AUC: {auc_score:.3f}")
    print(f"Specificity (SPE): {SPE:.3f}")
    print(f"Sensitivity (SEN): {SEN:.3f}")
    print(f"Negative Predictive Value (NPV): {NPV:.3f}")
    print(f"Positive Predictive Value (PPV): {PPV:.3f}")
    print(f"Matthews Correlation Coefficient (MCC): {MCC:.3f}")

    # Matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Predicted: 0", "Predicted: 1"],
                yticklabels=["Actual: 0", "Actual: 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix-tdmfgamTransformer.png", dpi=300)
    plt.show()

    # 6️⃣ Courbe ROC
    from sklearn.metrics import auc

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], '--', lw=1, label='Random Guess')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve_tdmfgamTransformer.png", dpi=300)
    plt.show()

    # ... après la partie ROC ...

    import numpy as np
    import torch

    # 1. Utilisation du GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    import numpy as np
    import torch
    from lime.lime_tabular import LimeTabularExplainer
    import matplotlib.pyplot as plt

    # 1. Définition des utilitaires pour concaténer/déconcaténer les features

    def make_global_vector(smiles, f, f1, f2, f3, f4, X_, A_):
        """Concatène toutes les entrées d'une molécule en un seul vecteur plat"""
        return np.concatenate([
            smiles.flatten(), 
            f.flatten(), 
            f1.flatten(), 
            f2.flatten(), 
            f3.flatten(), 
            f4.flatten(), 
            X_.flatten(), 
            A_.flatten()
        ])

    def split_global_vector(global_vector, shapes):
        """Découpe le vecteur global en chaque modalité d'entrée selon leurs shapes"""
        slices = np.cumsum([0] + [np.prod(s) for s in shapes])
        parts = [global_vector[slices[i]:slices[i+1]].reshape(shapes[i]) for i in range(len(shapes))]
        return parts  # [smiles, f, f1, f2, f3, f4, X, A]

    # 2. Calcul des shapes pour chaque entrée (adapte selon ton modèle !)
    test_idx = 0
    shapes = [
        smiles_tokens[test_idx:test_idx+1].shape[1:],   # (L,)
        mogen_fp[test_idx:test_idx+1].shape[1:],        # (d,)
        fp1[test_idx:test_idx+1].shape[1:],
        fp2[test_idx:test_idx+1].shape[1:],
        fp3[test_idx:test_idx+1].shape[1:],
        fp4[test_idx:test_idx+1].shape[1:],
        X[test_idx:test_idx+1].shape[1:],               # (N,N)
        A[test_idx:test_idx+1].shape[1:],               # (N,N)
    ]

    # 3. Création du background pour LIME
    background_size = 300
    background_vectors = np.array([
        make_global_vector(
            smiles_tokens[i], mogen_fp[i], fp1[i], fp2[i], fp3[i], fp4[i], X[i], A[i]
        ) for i in range(0, background_size)
    ])

    # 4. Sélection d'un échantillon du test à expliquer
    sample_idx = split_idx  # index du premier test
    to_explain_vector = make_global_vector(
        smiles_tokens[sample_idx], mogen_fp[sample_idx], fp1[sample_idx], fp2[sample_idx], fp3[sample_idx], fp4[sample_idx], X[sample_idx], A[sample_idx]
    )

    # 5. Définition du prédicteur LIME SANS erreur
    def lime_predict(X_vecs):
        results = []
        for vec in X_vecs:
            smiles, f, f1, f2, f3, f4, Xmat, Amat = split_global_vector(vec, shapes)
            # Clamp SMILES pour rester dans le vocabulaire de l'embedding
            vocab_size = 5000  # adapte selon ton vocab_size réel !
            smiles = np.clip(smiles, 0, vocab_size - 1)
            smiles_tensor = torch.LongTensor(smiles[np.newaxis, :]).to(device)
            f_tensor = torch.FloatTensor(f[np.newaxis, :]).to(device)
            f1_tensor = torch.FloatTensor(f1[np.newaxis, :]).to(device)
            f2_tensor = torch.FloatTensor(f2[np.newaxis, :]).to(device)
            f3_tensor = torch.FloatTensor(f3[np.newaxis, :]).to(device)
            f4_tensor = torch.FloatTensor(f4[np.newaxis, :]).to(device)
            Xmat_tensor = torch.FloatTensor(Xmat[np.newaxis, :, :]).to(device)
            Amat_tensor = torch.FloatTensor(Amat[np.newaxis, :, :]).to(device)
            label_ = torch.zeros(1).to(device)
            model.eval()
            with torch.no_grad():
                pred, _ = model(smiles_tensor, f_tensor, f1_tensor, f2_tensor, f3_tensor, f4_tensor, Xmat_tensor, Amat_tensor, label_)
            # Retourne [proba_0, proba_1] (LIME veut ce format !)
            pred_value = pred.cpu().numpy()[0]
            results.append([1 - pred_value, pred_value])
        return np.array(results)

    # 6. Générer les noms de features (pour visualisation)
    def make_feature_names():
        names = []
        for i in range(shapes[0][0]): names.append(f"SMILES_{i}")
        for i in range(shapes[1][0]): names.append(f"MOGEN_{i}")
        for i in range(shapes[2][0]): names.append(f"FP1_{i}")
        for i in range(shapes[3][0]): names.append(f"FP2_{i}")
        for i in range(shapes[4][0]): names.append(f"FP3_{i}")
        for i in range(shapes[5][0]): names.append(f"FP4_{i}")
        for i in range(shapes[6][0]): 
            for j in range(shapes[6][1]): names.append(f"GRAPH_X_{i}_{j}")
        for i in range(shapes[7][0]): 
            for j in range(shapes[7][1]): names.append(f"GRAPH_A_{i}_{j}")
        return names

    feature_names = make_feature_names()

    # 7. Création de l'explainer LIME
    explainer = LimeTabularExplainer(
        background_vectors,
        feature_names=feature_names,
        mode='classification',
        discretize_continuous=False
    )

    # 8. Génération de l'explication LIME pour un échantillon
    exp = explainer.explain_instance(
        to_explain_vector,
        lime_predict,
        num_features=50,   # nombre de features affichées
        top_labels=1
    )

    # 9. Affichage des résultats
    exp.show_in_notebook(show_table=True)

    fig = exp.as_pyplot_figure(label=1)
    fig.suptitle("LIME Explanation for Test Sample")
    fig.tight_layout()
    plt.show()
    fig.savefig("lime_explanation_sampleTranformer.png")