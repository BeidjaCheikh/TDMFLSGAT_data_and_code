# utils.py
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import torch
import warnings
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
from tokenizers import ByteLevelBPETokenizer  # Pour la tokenisation des SMILES

warnings.filterwarnings('ignore')

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GraphUtils():
    """Utility class for processing molecular graphs."""
    def _convert_smile_to_graph(self, smiles):
        features = []
        adj = []
        maxNumAtoms = 100
        for smile in smiles:
            iMol = Chem.MolFromSmiles(smile)
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)

            iFeature = np.zeros((maxNumAtoms, 65))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append(self.atom_feature(atom))
            iFeature[0:len(iFeatureTmp), 0:65] = iFeatureTmp
            iFeature = normalize(iFeature)

            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            iAdj = normalize_adj(iAdj)

            features.append(iFeature)
            adj.append(iAdj.A)
        features = np.asarray(features)
        adj = np.asarray(adj)
        return features, adj

    def atom_feature(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),
                                                   ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                                    'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                                    'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                                    'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                        self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                        self.one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                        [atom.GetIsAromatic()] + self.get_ring_info(atom))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def get_ring_info(self, atom):
        ring_info_feature = []
        for i in range(3, 9):
            ring_info_feature.append(1 if atom.IsInRingSize(i) else 0)
        return ring_info_feature

    def preprocess_smile(self, smiles):
        X, A = self._convert_smile_to_graph(smiles)
        return [X, A]

class SmilesUtils():
    """Utility class for SMILES tokenization."""
    def __init__(self, vocab_size=5000, max_len=79):
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size = vocab_size
        self.max_len = max_len

    def train_tokenizer(self, smiles_list):
        self.tokenizer.train_from_iterator(smiles_list, vocab_size=self.vocab_size)

    def tokenize_smiles(self, smiles_list):
        encoded = self.tokenizer.encode_batch(smiles_list)
        token_ids = [e.ids for e in encoded]
        padded = np.zeros((len(token_ids), self.max_len), dtype=np.int64)
        for i, seq in enumerate(token_ids):
            length = min(len(seq), self.max_len)
            padded[i, :length] = seq[:length]
        return torch.tensor(padded)

def load_data_with_smiles(path, smiles_utils):
    data = pd.read_csv(path)
    smiles = list(data['smiles'])
    labels = list(data['labels'])
    utils = GraphUtils()
    X, A = utils.preprocess_smile(smiles)
    smiles_tokens = smiles_utils.tokenize_smiles(smiles)
    fs = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fs.append(np.array(fp))
    mogen_fp = np.array(fs)
    f_sum = mogen_fp.sum(axis=-1)
    mogen_fp = mogen_fp / (np.reshape(f_sum, (-1, 1)))
    return X, A, mogen_fp, labels, smiles_tokens

def load_fp(path):
    data = pd.read_csv(path)
    features = data.values
    f_sum = features.sum(axis=-1)
    features = features / (np.reshape(f_sum, (-1, 1)))
    return features

class MyDataset(Dataset):
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
        return (self.smiles_tokens[index], self.f[index], self.f1[index], self.f2[index],
                    self.f3[index], self.f4[index], self.X[index], self.A[index], self.label[index])
