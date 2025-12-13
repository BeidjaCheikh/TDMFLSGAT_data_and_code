# interpretability_FigureX.py

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Draw

from lime.lime_tabular import LimeTabularExplainer

# Adapter le chemin à ton projet
sys.path.append(r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data')

from utils import (
    load_data_with_smiles,
    set_seed,
    load_fp,
    SmilesUtils,
    MyDataset
)
from DMFLSGAT_VZ import MyModel


###############################################
# 1. Chargement données + modèle entraîné
###############################################

SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# --- Chemins dataset internes (les mêmes que main_VZ.py) ---
smiles_csv_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/Smiles.csv'

PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/PubchemFP881.csv'
Topological_torsion_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/topological_torsion_fingerprints.csv'
APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/APC2D780.csv'
KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/FPs/KR.csv'

# --- Lire SMILES & labels ---
import pandas as pd
data = pd.read_csv(smiles_csv_path)
smiles_list = data['smiles'].tolist()

smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
smiles_utils.train_tokenizer(smiles_list)

X_np, A_np, mogen_fp_np, labels_np, smiles_tokens_np = load_data_with_smiles(smiles_csv_path, smiles_utils)

# Fingerprints additionnels
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
labels = torch.FloatTensor(labels_np)

# Même split que ton training
split_idx = 8284   # train : 0..8283, test : 8284..

# --- Modèle ---
model = MyModel(vocab_size=5000).to(device)
state_dict = torch.load("tdmflsgat_internal_best.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("✅ Modèle chargé.")


###############################################
# 2. Fonctions utilitaires pour les figures
###############################################

def plot_token_attention(attn_tokens, smiles_tokens_sample, out_png):
    """
    attn_tokens : (1, num_heads, L, L)
    """
    attn_mean = attn_tokens.mean(dim=1).squeeze(0).cpu().numpy()  # (L, L)

    token_row = smiles_tokens_sample[0].cpu().numpy()
    valid_len = int((token_row != 0).sum())
    if valid_len == 0:
        valid_len = attn_mean.shape[0]

    attn_trim = attn_mean[:valid_len, :valid_len]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        attn_trim,
        cmap="viridis",
        xticklabels=list(range(valid_len)),
        yticklabels=list(range(valid_len)),
        cbar_kws={"label": "Attention weight"}
    )
    plt.xlabel("Token index")
    plt.ylabel("Token index")
    plt.title("Figure X(A) – Token-level Transformer attention")
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()
    print("💾 Figure X(A) sauvegardée sous :", out_png)


def floyd_warshall_dist(adj_unweighted):
    """
    adj_unweighted : matrice d’adjacence bool/0-1 (N,N)
    -> distance minimale (N,N)
    """
    N = adj_unweighted.shape[0]
    dist = np.where(adj_unweighted > 0, 1.0, np.inf)
    np.fill_diagonal(dist, 0.0)
    for k in range(N):
        dist = np.minimum(dist, dist[:, k][:, None] + dist[k, :][None, :])
    return dist


def plot_atom_attention_radii(attn_graph_trim, mol, out_prefix):
    """
    attn_graph_trim : (num_atoms, num_atoms)
    mol : molécule RDKit
    """
    num_atoms = mol.GetNumAtoms()

    # adjacency non normalisée (0/1) depuis RDKit
    from rdkit.Chem import rdmolops
    adj_un = rdmolops.GetAdjacencyMatrix(mol)  # (N,N)
    dist = floyd_warshall_dist(adj_un)

    # rayon 1, 2, 3
    mats = {}
    for r in [1, 2, 3]:
        mask = (dist == r)
        mat_r = np.where(mask, attn_graph_trim, np.nan)  # les autres en NaN (blanc)
        mats[r] = mat_r

    vmin = np.nanmin(attn_graph_trim)
    vmax = np.nanmax(attn_graph_trim)

    # (a)(b)(c) : radius 1,2,3
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, r in zip(axes, [1, 2, 3]):
        sns.heatmap(
            mats[r],
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            xticklabels=list(range(num_atoms)),
            yticklabels=list(range(num_atoms)),
            cbar=False
        )
        ax.set_title(f"(Radius = {r})")
        ax.set_xlabel("Atom index")
        ax.set_ylabel("Atom index")
    fig.suptitle("Figure X(B) – Atom-level LSGAT attention by radius", y=1.02)
    plt.tight_layout()
    fig.savefig(out_prefix + "_radii123.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("💾 Figure X(B) (radius 1/2/3) sauvegardée.")

    # (d) matrice complète avec valeurs annotées
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        attn_graph_trim,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=list(range(num_atoms)),
        yticklabels=list(range(num_atoms)),
        cbar_kws={"label": "Attention weight"},
        annot=True,
        fmt=".2f"
    )
    plt.xlabel("Atom index")
    plt.ylabel("Atom index")
    plt.title("Figure X(B) – Atom-level LSGAT attention (full)")
    plt.tight_layout()
    plt.savefig(out_prefix + "_full.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("💾 Figure X(B) (full) sauvegardée.")


def plot_molecule_highlight(mol, atom_importance, top_k, out_png):
    num_atoms = mol.GetNumAtoms()
    top_k = min(top_k, num_atoms)
    top_idx = np.argsort(atom_importance)[-top_k:]
    print("Top atoms (indices RDKit) :", top_idx)

    # dessin publication-ready
    d = Draw.rdMolDraw2D.MolDraw2DCairo(600, 250)
    rdkit_mol = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(rdkit_mol)

    atom_highlight = {int(i): float(atom_importance[i]) for i in top_idx}
    max_imp = max(atom_highlight.values()) if atom_highlight else 1.0
    atom_radii = {i: 0.3 + 0.4 * (w / max_imp) for i, w in atom_highlight.items()}

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        rdkit_mol,
        highlightAtoms=list(atom_highlight.keys()),
        highlightAtomColors={i: (1.0, 0.5, 0.5) for i in atom_highlight.keys()},
        highlightAtomRadii=atom_radii
    )
    d.FinishDrawing()
    d.WriteDrawingText(out_png)
    print("💾 Figure X(C) sauvegardée sous :", out_png)


def make_global_fp_vector(f, f1, f2, f3, f4):
    return np.concatenate([f.flatten(), f1.flatten(), f2.flatten(), f3.flatten(), f4.flatten()])


def split_global_fp_vector(global_vector, shapes):
    slices = np.cumsum([0] + [np.prod(s) for s in shapes])
    parts = [global_vector[slices[i]:slices[i+1]].reshape(shapes[i]) for i in range(len(shapes))]
    return parts  # [f, f1, f2, f3, f4]


def make_fp_feature_names(shapes):
    names = []
    for i in range(shapes[0][0]): names.append(f"FP_Morgan_{i}")
    for i in range(shapes[1][0]): names.append(f"FP_PubChem_{i}")
    for i in range(shapes[2][0]): names.append(f"FP_TopoTorsion_{i}")
    for i in range(shapes[3][0]): names.append(f"FP_APC2D_{i}")
    for i in range(shapes[4][0]): names.append(f"FP_KR_{i}")
    return names


###############################################
# 3. Choix d’une molécule du test + forward
###############################################

# même choix que avant : première molécule du test set
sample_idx = split_idx
print("Sample index (test) :", sample_idx)

smiles_sample = smiles_list[sample_idx]
mol_sample = Chem.MolFromSmiles(smiles_sample)
num_atoms = mol_sample.GetNumAtoms()

print("Sample SMILES :", smiles_sample)
print("Nombre d'atomes :", num_atoms)

# batch=1
smiles_tokens_sample = smiles_tokens[sample_idx:sample_idx+1].to(device)
f_sample  = mogen_fp[sample_idx:sample_idx+1].to(device)
f1_sample = fp1[sample_idx:sample_idx+1].to(device)
f2_sample = fp2[sample_idx:sample_idx+1].to(device)
f3_sample = fp3[sample_idx:sample_idx+1].to(device)
f4_sample = fp4[sample_idx:sample_idx+1].to(device)
X_sample  = X[sample_idx:sample_idx+1].to(device)
A_sample  = A[sample_idx:sample_idx+1].to(device)
label_sample = labels[sample_idx:sample_idx+1].to(device)

# forward pour récupérer last_attention
with torch.no_grad():
    pred, _ = model(
        smiles_tokens_sample,
        f_sample,
        f1_sample,
        f2_sample,
        f3_sample,
        f4_sample,
        X_sample,
        A_sample,
        label_sample
    )
print("Prédiction hERG (proba blocage) :", float(pred.cpu().numpy()[0]))


###############################################
# 4. Figure X(A) – token attention
###############################################

with torch.no_grad():
    attn_tokens = model.smiles_transformer.get_token_attention(smiles_tokens_sample)

plot_token_attention(
    attn_tokens,
    smiles_tokens_sample,
    out_png="FigureX_A_token_attention_pub.png"
)


###############################################
# 5. Figure X(B) – graph attention (+radii)
###############################################

# On récupère la dernière couche LSGAT
last_lsgat_layer = model.graph_model.lsgat_layers[-1]

att_mats = []
for head in last_lsgat_layer:
    if head.last_attention is not None:
        att_mats.append(head.last_attention.cpu().numpy())

if len(att_mats) == 0:
    raise RuntimeError("Aucune matrice d'attention trouvée dans LSGATLayer.last_attention")

attn_graph_mean = np.mean(att_mats, axis=0)  # (Nmax, Nmax)
attn_graph_trim = attn_graph_mean[:num_atoms, :num_atoms]

plot_atom_attention_radii(
    attn_graph_trim,
    mol_sample,
    out_prefix="FigureX_B_atom_attention_pub"
)

# importance globale par atome
atom_importance = attn_graph_trim.mean(axis=1)


###############################################
# 6. Figure X(C) – molécule + atomes importants
###############################################

plot_molecule_highlight(
    mol_sample,
    atom_importance,
    top_k=8,
    out_png="FigureX_C_molecule_highlighted_pub.png"
)


###############################################
# 7. Figure X(D) – LIME sur fingerprints
###############################################

# shapes des FP
shapes = [
    (mogen_fp.shape[1],),
    (fp1.shape[1],),
    (fp2.shape[1],),
    (fp3.shape[1],),
    (fp4.shape[1],),
]

# background LIME (300 molécules)
background_size = min(300, mogen_fp.shape[0])
background_vectors = []

for i in range(background_size):
    vec = make_global_fp_vector(
        mogen_fp[i].cpu().numpy(),
        fp1[i].cpu().numpy(),
        fp2[i].cpu().numpy(),
        fp3[i].cpu().numpy(),
        fp4[i].cpu().numpy()
    )
    background_vectors.append(vec)

background_vectors = np.array(background_vectors)

# vecteur à expliquer = même molécule que (A,B,C)
to_explain_vector = make_global_fp_vector(
    mogen_fp[sample_idx].cpu().numpy(),
    fp1[sample_idx].cpu().numpy(),
    fp2[sample_idx].cpu().numpy(),
    fp3[sample_idx].cpu().numpy(),
    fp4[sample_idx].cpu().numpy()
)

def lime_predict_fp(X_vecs):
    results = []
    for vec in X_vecs:
        f_v, f1_v, f2_v, f3_v, f4_v = split_global_fp_vector(vec, shapes)

        f_t  = torch.FloatTensor(f_v[np.newaxis, :]).to(device)
        f1_t = torch.FloatTensor(f1_v[np.newaxis, :]).to(device)
        f2_t = torch.FloatTensor(f2_v[np.newaxis, :]).to(device)
        f3_t = torch.FloatTensor(f3_v[np.newaxis, :]).to(device)
        f4_t = torch.FloatTensor(f4_v[np.newaxis, :]).to(device)

        smiles_t = smiles_tokens_sample
        X_t      = X_sample
        A_t      = A_sample
        label_   = torch.zeros(1).to(device)

        model.eval()
        with torch.no_grad():
            pred, _ = model(
                smiles_t,
                f_t,
                f1_t,
                f2_t,
                f3_t,
                f4_t,
                X_t,
                A_t,
                label_
            )
        p = float(pred.cpu().numpy()[0])
        results.append([1 - p, p])
    return np.array(results)

feature_names = make_fp_feature_names(shapes)

explainer = LimeTabularExplainer(
    background_vectors,
    feature_names=feature_names,
    mode="classification",
    discretize_continuous=False
)

exp = explainer.explain_instance(
    to_explain_vector,
    lime_predict_fp,
    num_features=20,
    top_labels=1
)

fig = exp.as_pyplot_figure(label=1)
fig.suptitle("Figure X(D) – Fingerprint-based LIME explanation", y=1.02)
plt.tight_layout()
plt.savefig("FigureX_D_LIME_fp_pub.png", dpi=600, bbox_inches="tight")
plt.close()

print("✅ Toutes les sous-figures Figure X (A–D) publication-ready sont générées.")
