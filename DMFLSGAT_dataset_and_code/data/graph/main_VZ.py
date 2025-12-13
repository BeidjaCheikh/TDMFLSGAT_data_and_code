# main_VZ.py
import sys
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

import io
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
)

from utils import (
    load_data_with_smiles,
    set_seed,
    load_fp,
    SmilesUtils,
    MyDataset
)
from DMFLSGAT_VZ import MyModel

from rdkit import Chem
from rdkit.Chem import Draw
from lime.lime_tabular import LimeTabularExplainer

from PIL import Image, ImageDraw


def draw_molecule_with_orange_ellipse(mol, highlight_atoms, output_path):
    """
    Dessine la molécule avec atomes importants + ellipse orange transparente,
    dans le style de la figure 6 de hERGAT.
    """
    # Dessin RDKit en mode Cairo (PNG)
    width, height = 800, 400
    drawer = Draw.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False

    # Couleur de surlignage des atomes importants (rouge léger)
    highlight_color = (1.0, 0.0, 0.0, 0.6)

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=[int(i) for i in highlight_atoms],
        highlightAtomColors={int(i): highlight_color for i in highlight_atoms},
    )
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()

    # Image PIL
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    # Récupération des coordonnées (en pixels) des atomes importants
    xs, ys = [], []
    for idx in highlight_atoms:
        pt = drawer.GetDrawCoords(int(idx))  # coord. dessin
        xs.append(pt.x)
        ys.append(pt.y)

    if len(xs) > 0:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # petite marge autour du groupe d’atomes (en pixels)
        margin_x = 25
        margin_y = 25

        ellipse_box = [
            min_x - margin_x,
            min_y - margin_y,
            max_x + margin_x,
            max_y + margin_y,
        ]

        # Couleur orange pastel (style hERGAT)
        outline_color = (255, 160, 80, 220)   # contour orange
        fill_color = (255, 210, 150, 90)      # remplissage orange transparent

        draw.ellipse(
            ellipse_box,
            outline=outline_color,
            fill=fill_color,
            width=8,
        )

    img.save(output_path)
    print(f"✅ Figure X(C) avec ellipse orange sauvegardée : {output_path}")


if __name__ == '__main__':

    ############################################################
    # 0. Initialisation & chargement des données
    ############################################################
    SEED = 42
    set_seed(SEED)

    # --- Chargement CSV SMILES ---
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/Smiles.csv'
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()

    smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
    smiles_utils.train_tokenizer(smiles_list)

    X_np, A_np, mogen_fp_np, labels_np, smiles_tokens_np = load_data_with_smiles(
        path, smiles_utils
    )

    # --- Chemins FPs ---
    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/PubchemFP881.csv'
    Topological_torsion_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/topological_torsion_fingerprints.csv'
    APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/APC2D780.csv'
    KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/KR.csv'

    fp1_np = load_fp(PubchemFP881_path)
    fp2_np = load_fp(Topological_torsion_path)
    fp3_np = load_fp(APC2D780_path)
    fp4_np = load_fp(KR_path)

    # --- Conversion en tenseurs ---
    X = torch.FloatTensor(X_np)
    A = torch.FloatTensor(A_np)
    mogen_fp = torch.FloatTensor(mogen_fp_np)
    fp1 = torch.FloatTensor(fp1_np)
    fp2 = torch.FloatTensor(fp2_np)
    fp3 = torch.FloatTensor(fp3_np)
    fp4 = torch.FloatTensor(fp4_np)
    smiles_tokens = torch.LongTensor(smiles_tokens_np)
    labels = torch.FloatTensor(labels_np)

    ############################################################
    # 1. Split train / test + DataLoaders
    ############################################################
    split_idx = 8284  # même choix que tes scripts précédents

    train_dataset = MyDataset(
        smiles_tokens=smiles_tokens[:split_idx],
        f=mogen_fp[:split_idx],
        f1=fp1[:split_idx],
        f2=fp2[:split_idx],
        f3=fp3[:split_idx],
        f4=fp4[:split_idx],
        X=X[:split_idx],
        A=A[:split_idx],
        label=labels[:split_idx],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=100, shuffle=True, drop_last=True
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
        label=labels[split_idx:],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=100, shuffle=False, drop_last=True
    )

    ############################################################
    # 2. Modèle & optimiseur
    ############################################################
    model = MyModel(vocab_size=5000)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=150, eta_min=0
    )

    ############################################################
    # 3. Entraînement
    ############################################################
    model.train()
    num_epochs = 5
    all_epoch_losses = []

    for epoch in range(num_epochs):
        pred_y, PED, true_y = [], [], []
        epoch_losses = []

        for i, batch_data in enumerate(train_loader):
            (
                smiles_tokens_b,
                fp_b,
                fp1_b,
                fp2_b,
                fp3_b,
                fp4_b,
                X_b,
                A_b,
                label_b,
            ) = batch_data

            optimizer.zero_grad()
            logits, loss = model(
                smiles_tokens_b,
                fp_b,
                fp1_b,
                fp2_b,
                fp3_b,
                fp4_b,
                X_b,
                A_b,
                label_b,
            )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            batch_loss_value = loss.item()
            epoch_losses.append(batch_loss_value)
            print(
                f"Epoch: {epoch+1}/{num_epochs}, "
                f"Batch: {i+1}, Loss: {batch_loss_value:.4f}"
            )

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
    plt.plot(
        range(1, num_epochs + 1),
        all_epoch_losses,
        marker='o',
        label='Loss moyenne par epoch',
    )
    plt.title("Courbe de la Loss durant l'entraînement")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_loss_curve_tdmflsgat.png", dpi=300)
    plt.show()

    ############################################################
    # 4. Évaluation sur le test
    ############################################################
    model.eval()
    pred_y, true_y = [], []
    all_labels, all_probs = [], []

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            (
                smiles_tokens_b,
                fp_b,
                fp1_b,
                fp2_b,
                fp3_b,
                fp4_b,
                X_b,
                A_b,
                label_b,
            ) = batch_data

            logits, loss = model(
                smiles_tokens_b,
                fp_b,
                fp1_b,
                fp2_b,
                fp3_b,
                fp4_b,
                X_b,
                A_b,
                label_b,
            )
            logits_np = logits.detach().cpu().numpy()
            pred_y.extend(logits_np)
            true_y.extend(label_b.cpu().numpy())
            all_labels.extend(label_b.cpu().numpy())
            all_probs.extend(logits_np)

    # Distribution des probabilités
    plt.figure(figsize=(6, 4))
    plt.hist(pred_y, bins=50)
    plt.title('Distribution des probabilités prédites')
    plt.xlabel('Probabilité')
    plt.ylabel('Nombre')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prob_distribution_tdmflsgat.png", dpi=300)
    plt.show()

    # Seuil optimal via ROC
    fpr, tpr, thresholds = roc_curve(true_y, pred_y)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Seuil optimal trouvé via ROC : {optimal_threshold:.4f}")

    PED = (np.array(pred_y) >= optimal_threshold).astype(int)

    cm = confusion_matrix(true_y, PED)
    try:
        TN, FP, FN, TP = cm.ravel()
    except Exception as e:
        print("Erreur dans la matrice de confusion :", e)
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

    print('\n========== Évaluation sur Test Set ==========')
    print(f"Test Accuracy (ACC): {acc:.3f}")
    print(f"Test AUC: {auc_score:.3f}")
    print(f"Specificity (SPE): {SPE:.3f}")
    print(f"Sensitivity (SEN): {SEN:.3f}")
    print(f"Negative Predictive Value (NPV): {NPV:.3f}")
    print(f"Positive Predictive Value (PPV): {PPV:.3f}")
    print(f"Matthews Correlation Coefficient (MCC): {MCC:.3f}")

    torch.save(model.state_dict(), "tdmflsgat_internal_best.pth")
    print("✅ Modèle sauvegardé sous : tdmflsgat_internal_best.pth")

    # Matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=["Predicted: 0", "Predicted: 1"],
        yticklabels=["Actual: 0", "Actual: 1"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_tdmflsgat.png", dpi=300)
    plt.show()

    # ROC Curve
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
    plt.savefig("roc_curve_tdmflsgat.png", dpi=300)
    plt.show()

    ############################################################
    # 5. Device pour interprétabilité
    ############################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ############################################################
    # 6. Choix d'une molécule représentative pour Figure X
    ############################################################
    all_labels_arr = np.array(all_labels)
    all_probs_arr = np.array(all_probs)

    blocker_idx_local = np.where(all_labels_arr == 1)[0]
    if len(blocker_idx_local) > 0:
        best_blocker_local = blocker_idx_local[np.argmax(all_probs_arr[blocker_idx_local])]
        sample_idx = split_idx + best_blocker_local
    else:
        sample_idx = split_idx

    smiles_sample = smiles_list[sample_idx]
    mol_sample = Chem.MolFromSmiles(smiles_sample)
    num_atoms = mol_sample.GetNumAtoms()

    print("Sample SMILES:", smiles_sample)
    print("Nombre d'atomes:", num_atoms)

    # Tenseurs batch=1 pour cet échantillon
    smiles_tokens_sample = smiles_tokens[sample_idx:sample_idx + 1].to(device)
    fp_b_sample = mogen_fp[sample_idx:sample_idx + 1].to(device)
    fp1_b_sample = fp1[sample_idx:sample_idx + 1].to(device)
    fp2_b_sample = fp2[sample_idx:sample_idx + 1].to(device)
    fp3_b_sample = fp3[sample_idx:sample_idx + 1].to(device)
    fp4_b_sample = fp4[sample_idx:sample_idx + 1].to(device)
    X_b_sample = X[sample_idx:sample_idx + 1].to(device)
    A_b_sample = A[sample_idx:sample_idx + 1].to(device)
    label_b_sample = labels[sample_idx:sample_idx + 1].to(device)

    ############################################################
    # 7. Figure X(A) – attention du Transformer sur les tokens
    ############################################################
    model.eval()
    with torch.no_grad():
        attn_tokens = model.smiles_transformer.get_token_attention(
            smiles_tokens_sample
        )  # (1, num_heads, L, L)

    attn_tokens_mean = attn_tokens.mean(dim=1).squeeze(0).cpu().numpy()  # (L, L)

    token_row = smiles_tokens_sample[0].cpu().numpy()
    valid_len = int((token_row != 0).sum())
    if valid_len == 0:
        valid_len = attn_tokens_mean.shape[0]

    attn_tokens_trim = attn_tokens_mean[:valid_len, :valid_len]

    vmin = np.percentile(attn_tokens_trim, 5)
    vmax = np.percentile(attn_tokens_trim, 95)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_tokens_trim, vmin=vmin, vmax=vmax, aspect='equal')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention weight", fontsize=11)

    ax.set_xlabel("Token index", fontsize=11)
    ax.set_ylabel("Token index", fontsize=11)

    ax.set_xticks(range(valid_len))
    ax.set_yticks(range(valid_len))
    ax.set_xticklabels(range(valid_len), fontsize=6, rotation=90)
    ax.set_yticklabels(range(valid_len), fontsize=6)

    ax.tick_params(length=2, pad=2)
    ax.set_xlim(-0.5, valid_len - 1 + 0.5)
    ax.set_ylim(valid_len - 1 + 0.5, -0.5)

    fig.tight_layout()
    fig.savefig("FigureX_A_token_attention_pub.png", dpi=300)
    plt.close(fig)

    ############################################################
    # 8. Figure X(B,C) – attention LSGAT & atomes importants
    ############################################################
    with torch.no_grad():
        _ = model(
            smiles_tokens_sample,
            fp_b_sample,
            fp1_b_sample,
            fp2_b_sample,
            fp3_b_sample,
            fp4_b_sample,
            X_b_sample,
            A_b_sample,
            label_b_sample,
        )

    last_lsgat_layer = model.graph_model.lsgat_layers[-1]

    att_mats = []
    for head in last_lsgat_layer:
        if head.last_attention is not None:
            att_mats.append(head.last_attention.cpu().numpy())

    if len(att_mats) > 0:
        attn_graph_mean = np.mean(att_mats, axis=0)  # (N, N)
        attn_graph_trim = attn_graph_mean[:num_atoms, :num_atoms]

        # normalisation [0,1] pour une belle échelle de couleur
        attn_graph_norm = attn_graph_trim / (attn_graph_trim.max() + 1e-8)

        # ---------- Figure X(B) : heatmap complète ----------
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_graph_norm, vmin=0.0, vmax=1.0, aspect='equal')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Attention weight", fontsize=11)

        ax.set_xlabel("Atom index", fontsize=11)
        ax.set_ylabel("Atom index", fontsize=11)

        ax.set_xticks(range(num_atoms))
        ax.set_yticks(range(num_atoms))
        ax.set_xticklabels(range(num_atoms), fontsize=6, rotation=90)
        ax.set_yticklabels(range(num_atoms), fontsize=6)

        ax.tick_params(length=2, pad=2)
        ax.set_xlim(-0.5, num_atoms - 1 + 0.5)
        ax.set_ylim(num_atoms - 1 + 0.5, -0.5)

        fig.tight_layout()
        fig.savefig("FigureX_B_atom_attention_pub_full.png", dpi=300)
        plt.close(fig)

        # ---------- Importance moyenne par atome ----------
        atom_importance = attn_graph_norm.mean(axis=1)  # (num_atoms,)

    else:
        print("⚠️ Aucune attention capturée dans la dernière couche LSGAT.")
        atom_importance = np.zeros(num_atoms)

    # ---------- Figure X(C) : atomes les plus importants sur la molécule ----------
    top_k = min(8, num_atoms)
    top_atoms = np.argsort(atom_importance)[-top_k:]
    print("Indices des atomes les plus importants:", top_atoms)

    # RDKit : image large + atomes surlignés
    img = Draw.MolToImage(
        mol_sample,
        size=(900, 300),
        highlightAtoms=[int(i) for i in top_atoms],
    )
    img.save("FigureX_C_molecule_highlighted_pub.png")

    print("Images Figure X (A,B,C) générées (version publication).")


    ############################################################
    # 9. Figure X(D) – LIME sur les fingerprints uniquement
    ############################################################
    def make_global_vector(f, f1, f2, f3, f4):
        return np.concatenate(
            [f.flatten(), f1.flatten(), f2.flatten(), f3.flatten(), f4.flatten()]
        )

    def split_global_vector(global_vector, shapes):
        slices = np.cumsum([0] + [np.prod(s) for s in shapes])
        parts = [
            global_vector[slices[i]:slices[i + 1]].reshape(shapes[i])
            for i in range(len(shapes))
        ]
        return parts  # [f, f1, f2, f3, f4]

    shapes = [
        (mogen_fp.shape[1],),
        (fp1.shape[1],),
        (fp2.shape[1],),
        (fp3.shape[1],),
        (fp4.shape[1],),
    ]

    # Background LIME
    background_size = min(300, mogen_fp.shape[0])
    background_vectors = []
    for i in range(background_size):
        f_bg = mogen_fp[i].cpu().numpy()
        f1_bg = fp1[i].cpu().numpy()
        f2_bg = fp2[i].cpu().numpy()
        f3_bg = fp3[i].cpu().numpy()
        f4_bg = fp4[i].cpu().numpy()
        background_vectors.append(
            make_global_vector(f_bg, f1_bg, f2_bg, f3_bg, f4_bg)
        )
    background_vectors = np.array(background_vectors)

    # FP de la molécule sample
    f_sample = mogen_fp[sample_idx].cpu().numpy()
    f1_sample = fp1[sample_idx].cpu().numpy()
    f2_sample = fp2[sample_idx].cpu().numpy()
    f3_sample = fp3[sample_idx].cpu().numpy()
    f4_sample = fp4[sample_idx].cpu().numpy()

    to_explain_vector = make_global_vector(
        f_sample, f1_sample, f2_sample, f3_sample, f4_sample
    )

    def lime_predict(X_vecs):
        results = []
        for vec in X_vecs:
            f_v, f1_v, f2_v, f3_v, f4_v = split_global_vector(vec, shapes)

            f_tensor = torch.FloatTensor(f_v[np.newaxis, :]).to(device)
            f1_tensor = torch.FloatTensor(f1_v[np.newaxis, :]).to(device)
            f2_tensor = torch.FloatTensor(f2_v[np.newaxis, :]).to(device)
            f3_tensor = torch.FloatTensor(f3_v[np.newaxis, :]).to(device)
            f4_tensor = torch.FloatTensor(f4_v[np.newaxis, :]).to(device)

            smiles_tensor = smiles_tokens_sample
            Xmat_tensor = X_b_sample
            Amat_tensor = A_b_sample
            label_ = torch.zeros(1).to(device)

            model.eval()
            with torch.no_grad():
                pred, _ = model(
                    smiles_tensor,
                    f_tensor,
                    f1_tensor,
                    f2_tensor,
                    f3_tensor,
                    f4_tensor,
                    Xmat_tensor,
                    Amat_tensor,
                    label_,
                )
            pred_value = float(pred.cpu().numpy()[0])
            results.append([1.0 - pred_value, pred_value])
        return np.array(results)

    def make_feature_names():
        names = []
        for i in range(shapes[0][0]):
            names.append(f"FP_Morgan_{i}")
        for i in range(shapes[1][0]):
            names.append(f"FP_PubChem_{i}")
        for i in range(shapes[2][0]):
            names.append(f"FP_TopoTorsion_{i}")
        for i in range(shapes[3][0]):
            names.append(f"FP_APC2D_{i}")
        for i in range(shapes[4][0]):
            names.append(f"FP_KR_{i}")
        return names

    feature_names = make_feature_names()

    explainer = LimeTabularExplainer(
        background_vectors,
        feature_names=feature_names,
        mode='classification',
        discretize_continuous=False,
    )

    exp = explainer.explain_instance(
        to_explain_vector,
        lime_predict,
        num_features=20,
        top_labels=1,
    )

    fig = exp.as_pyplot_figure(label=1)
    fig.tight_layout()
    fig.savefig("FigureX_D_LIME_fp_pub.png", dpi=300)
    plt.close(fig)

    print("✅ Figure X (A, B, C, D) générée (version publication).")
