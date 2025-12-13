# utils.py

import sys
import random
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from tokenizers import ByteLevelBPETokenizer  # Pour la tokenisation des SMILES
import numpy as np
import logging
from torch.utils.data import DataLoader, Subset

# Désactiver les warnings RDKit
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')


def normalize(mx):
    """
    Normalisation ligne par ligne d'une matrice (type SciPy sparse ou ndarray).
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """
    Normalisation symétrique de la matrice d'adjacence.
    Retourne D^-0.5 A D^-0.5 au format COO.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    # D^-0.5 A D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class GraphUtils:
    """
    Utility class pour la construction des graphes moléculaires.
    """

    def _convert_smile_to_graph(self, smiles):
        features = []
        adj = []
        maxNumAtoms = 100

        for smile in smiles:
            iMol = Chem.MolFromSmiles(smile)
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)

            # Matrice de features (maxNumAtoms x 65)
            iFeature = np.zeros((maxNumAtoms, 65))
            iFeatureTmp = []

            for atom in iMol.GetAtoms():
                iFeatureTmp.append(self.atom_feature(atom))

            iFeature[0:len(iFeatureTmp), 0:65] = iFeatureTmp
            iFeature = normalize(iFeature)

            # Matrice d’adjacence + self-loop
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = (
                iAdjTmp + np.eye(len(iFeatureTmp))
            )
            iAdj = normalize_adj(iAdj)

            features.append(iFeature)
            adj.append(iAdj.A)

        features = np.asarray(features)
        adj = np.asarray(adj)

        return features, adj

    def atom_feature(self, atom):
        """
        Feature vector d’un atome (taille 65).
        """
        return np.array(
            self.one_of_k_encoding_unk(
                atom.GetSymbol(),
                [
                    'C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                    'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I',
                    'B', 'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                    'Ti', 'Zn', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr',
                    'Pt', 'Hg', 'Pb'
                ]
            )
            + self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + self.one_of_k_encoding(
                atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]
            )
            + [atom.GetIsAromatic()]
            + self.get_ring_info(atom)
        )

    def one_of_k_encoding_unk(self, x, allowable_set):
        """
        One-hot; mappe les valeurs inconnues sur le dernier élément.
        """
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding(self, x, allowable_set):
        """
        One-hot strict : lève une exception si x n'est pas dans allowable_set.
        """
        if x not in allowable_set:
            raise Exception(
                "input {0} not in allowable set{1}:".format(x, allowable_set)
            )
        return list(map(lambda s: x == s, allowable_set))

    def get_ring_info(self, atom):
        """
        Indique si l'atome appartient à des cycles de taille 3 à 8.
        """
        ring_info_feature = []
        for i in range(3, 9):
            ring_info_feature.append(1 if atom.IsInRingSize(i) else 0)
        return ring_info_feature

    def preprocess_smile(self, smiles):
        """
        Transforme une liste de SMILES en (X, A) pour le GNN.
        """
        X, A = self._convert_smile_to_graph(smiles)
        return [X, A]


class SmilesUtils:
    """
    Utility class pour la tokenisation des SMILES avec Byte-Level BPE.
    """

    def __init__(self, vocab_size=5000, max_len=79):
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size = vocab_size
        self.max_len = max_len

    def train_tokenizer(self, smiles_list):
        """
        Entraîne le tokenizer sur une liste de SMILES.
        """
        self.tokenizer.train_from_iterator(
            smiles_list, vocab_size=self.vocab_size
        )

    def tokenize_smiles(self, smiles_list):
        """
        Tokenise et pad une liste de SMILES en tenseur (N, max_len).
        """
        encoded = self.tokenizer.encode_batch(smiles_list)
        token_ids = [e.ids for e in encoded]

        padded = np.zeros((len(token_ids), self.max_len), dtype=np.int64)
        for i, seq in enumerate(token_ids):
            length = min(len(seq), self.max_len)
            padded[i, :length] = seq[:length]

        return torch.tensor(padded)


def load_data_with_smiles(path, smiles_utils):
    """
    Charge un CSV contenant les colonnes 'smiles' et 'labels'
    et retourne :
      X : features des graphes (N, maxAtoms, 65)
      A : adjacences normalisées (N, maxAtoms, maxAtoms)
      mogen_fp : Morgan FP normalisés (N, 1024)
      labels : liste ou array des labels
      smiles_tokens : token IDs (N, max_len)
    """
    data = pd.read_csv(path)
    smiles = list(data['smiles'])
    labels = list(data['labels'])

    utils = GraphUtils()
    X, A = utils.preprocess_smile(smiles)

    # Tokenisation des SMILES
    smiles_tokens = smiles_utils.tokenize_smiles(smiles)

    # Morgan fingerprints
    fs = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fs.append(np.array(fp))

        mogen_fp = np.array(fs, dtype=float)
        f_sum = mogen_fp.sum(axis=-1)
        f_sum[f_sum == 0] = 1.0          # <-- important
        mogen_fp = mogen_fp / f_sum[:, None]

    return X, A, mogen_fp, labels, smiles_tokens

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}
    formatter = logging.Formatter("[%(asctime)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # File
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# ---------------------------------------------------------
# 4. K-fold split adapté à TON dataset (MyDataset)
# ---------------------------------------------------------
def load_fold_data_tdmf(at_fold, batch, cpus_per_gpu, k, dataset):
    """
    Réplique la logique de hERG-MFFGNN :
      - On garde le dernier "fold" comme test fixe
      - Les k folds restants tournent comme validation
    dataset : instance de MyDataset
    """
    N = len(dataset)
    folds = k + 1          # même logique : k folds val + 1 fold test
    fold_size = N // folds
    test_start = N - fold_size

    val_start = at_fold * fold_size

    test_indices = list(range(test_start, N))

    if at_fold != k - 1:
        val_end = (at_fold + 1) * fold_size
        val_indices = list(range(val_start, val_end))
        train_indices = list(range(0, val_start)) + list(range(val_end, test_start))
    else:
        val_indices = list(range(val_start, test_start))
        train_indices = list(range(0, val_start))

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, val_indices)
    test_set  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                              pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=batch, shuffle=False,
                              pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch, shuffle=False,
                              pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)

    return train_loader, valid_loader, test_loader


def load_fp(path):
    """
    Charge un CSV de descripteurs/fingerprints et les normalise par la somme,
    en évitant les divisions par zéro.
    """
    data = pd.read_csv(path)
    features = data.values.astype(float)

    f_sum = features.sum(axis=-1)
    # Éviter f_sum = 0
    f_sum[f_sum == 0] = 1.0

    features = features / f_sum[:, None]
    return features



class MyDataset(Dataset):
    """
    Dataset PyTorch pour combiner :
      - tokens SMILES
      - fingerprint global f
      - autres fingerprints f1..f4
      - graph features X, A
      - labels
    """

    def __init__(self, smiles_tokens, f, f1, f2, f3, f4, X, A, label):
        self.smiles_tokens = smiles_tokens
        self.f = f
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.X = X
        self.A = A
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return (
            self.smiles_tokens[index],
            self.f[index],
            self.f1[index],
            self.f2[index],
            self.f3[index],
            self.f4[index],
            self.X[index],
            self.A[index],
            self.label[index],
        )

