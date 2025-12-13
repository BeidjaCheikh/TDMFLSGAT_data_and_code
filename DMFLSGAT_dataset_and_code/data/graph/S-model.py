############################################
# model.py
############################################

import sys
sys.path.append('/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data')

import torch
import torch.nn as nn
import torch.nn.functional as F
from graph.layer import GraphAttentionLayer  # Veillez à disposer de ce module

#############################################################
# 1) SmilesTransformer: encodeur Transformer basique
#############################################################
class SmilesTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=256, num_layers=6, max_len=79, dropout=0.2):
        super(SmilesTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.normal_(self.positional_encoding, std=0.02)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Ajoute une LayerNorm avant l'opération
            ),
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_tokens):
        # (batch, seq_len)
        x = self.embedding(smiles_tokens)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x)
        # Pooling: mean + max concat
        x_mean = x.mean(dim=1)
        x_max, _ = x.max(dim=1)
        out = torch.cat([x_mean, x_max], dim=-1)
        return out  # (batch, embed_dim * 2)


#############################################################
# 2) GCN basique
#############################################################
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        # Ex: transforme 65 -> 65
        self.weight = nn.Parameter(torch.randn(65, 65), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(65), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        """
        Param inputs: tuple (X, A)
          - X: (batch_size, maxNumAtoms=100, 65) => on traite en batch ou unitairement
          - A: (batch_size, maxNumAtoms=100, maxNumAtoms=100)
        """
        X, A = inputs
        xw = torch.matmul(X, self.weight)
        out = torch.matmul(A, xw)
        out += self.bias
        out = self.relu(out)
        return out, A


#############################################################
# 3) GraphModel: applique GAT + projection
#############################################################
class GraphModel(nn.Module):
    def __init__(self):
        super(GraphModel, self).__init__()
        # Nombre de têtes GAT
        self.num_head = 4

        # Option GCN si besoin (ici non utilisé dans l'exemple final)
        self.layers = nn.Sequential(
            GCN(),
            GCN(),
            GCN(),
         
        )

        # Projection finale après concat GAT
        self.proj = nn.Sequential(
            nn.Linear(32*4*100, 1024),  # => (batch_size, 32*4*100) -> 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Couche GAT
        self.att = GraphAttentionLayer()

    def forward(self, X, A):
        """
        Param X: (batch_size, 100, 32) ou (batch_size, 100, 65) => dans votre code, x a 65 features
        Param A: (batch_size, 100, 100)
        Retourne un tenseur (batch_size, 128)
        """
        # EXEMPLE GCN (commenté)
        # out = self.layers((X, A))[0]

        # GAT
        features = []
        # On traite chaque élément du batch séparément
        for i in range(X.shape[0]):
            feature_temp = []
            x, a = X[i], A[i]  # => (100, 65), (100, 100)
            # 2 couches GAT => ici c'est "num_head" boucles, pas forcément 2
            for _ in range(self.num_head):
                ax = self.att(x, a)  # => (100, 32)
                feature_temp.append(ax)
            # Concat sur la dimension features
            feature_temp = torch.cat(feature_temp, dim=1)  # => (100, 32 * num_head)
            features.append(feature_temp)

        out = torch.stack(features, dim=0)  # => (batch_size, 100, 32 * num_head)
        out = out.view(out.size(0), -1)     # => (batch_size, 100 * 32 * num_head)
        out = self.proj(out)                # => (batch_size, 128)

        return out


#############################################################
# 4) FpModel: projection multiple sur 5 fingerprints
#############################################################
class FpModel(nn.Module):
    def __init__(self):
        super(FpModel, self).__init__()

        # Morgan FP => (1024->512->256->128)
        self.fp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # PubChem 881 => (881->512->256->128)
        self.fp1 = nn.Sequential(
            nn.Linear(881, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Topological torsion 1024 => (1024->512->256->128)
        self.fp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # APC2D780 => (780->512->256->128)
        self.fp3 = nn.Sequential(
            nn.Linear(780, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # KR => (4860->512->256->128)
        self.fp4 = nn.Sequential(
            nn.Linear(4860, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, x, x1, x2, x3, x4):
        """
        Param x, x1, x2, x3, x4 : tenseurs (batch_size, dimension_fp)
        Retourne 5 tenseurs => (batch_size, 128) chacun
        """
        x  = self.fp(x)
        x1 = self.fp1(x1)
        x2 = self.fp2(x2)
        x3 = self.fp3(x3)
        x4 = self.fp4(x4)
        return x, x1, x2, x3, x4


#############################################################
# 5) MyModel: assemble SmilesTransformer + GraphModel + FpModel
#############################################################
class MyModel(nn.Module):
    def __init__(self, vocab_size):
        super(MyModel, self).__init__()
        self.smiles_transformer = SmilesTransformer(vocab_size)
        self.graph_model = GraphModel()
        self.fp_model = FpModel()

        self.proj = nn.Sequential(
            nn.Linear(256 + 128 + 128 * 5, 128),  # 256 (transformer) + 128 (graph) + 640 (fp)
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, 1)
        self.active = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, smiles_tokens, f, f1, f2, f3, f4, X, A, label):
        smiles_features = self.smiles_transformer(smiles_tokens)  # (batch, 256)
        graph_features = self.graph_model(X, A)                   # (batch, 128)
        fp_features = self.fp_model(f, f1, f2, f3, f4)            # tuple of 5 * (batch, 128)
        fp_concat = torch.cat(fp_features, dim=1)                 # (batch, 640)
        combined = torch.cat((smiles_features, graph_features, fp_concat), dim=1)  # (batch, 1024)
        x = self.proj(combined)                                   # (batch, 128)
        x = self.fc(x)                                            # (batch, 1)
        x = self.active(x).squeeze(-1)
        loss = self.loss_fn(x, label)
        return x, loss



#############################################################
# 6) Exemple d'utilisation dans le main
#############################################################
if __name__ == '__main__':
    # -------------------------------------------------------------
    # Supposons que vous disposiez de 'utils.py' avec :
    #   - set_seed
    #   - SmilesUtils
    #   - load_data_with_smiles
    #   - load_fp
    #   - etc.
    # Ici on montre juste un exemple illustratif :
    # -------------------------------------------------------------
    from graph.utils import load_data_with_smiles, set_seed, load_fp
    from torch.utils.data import DataLoader
    from utils import SmilesUtils
    import torch
    from torch.utils.data import Dataset
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    import pandas as pd

    # Définition d'un mini-dataset PyTorch
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
            return (self.smiles_tokens[index],
                    self.f[index], self.f1[index], self.f2[index],
                    self.f3[index], self.f4[index],
                    self.X[index], self.A[index],
                    self.label[index])

    # 1) Fixer la seed
    SEED = 42
    set_seed(SEED)

    # 2) Initialiser un tokenizer SMILES
    smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/Smiles.csv'

    # 3) Charger les données (et entraîner le tokenizer SMILES)
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()
    smiles_utils.train_tokenizer(smiles_list)

    # 4) Charger X, A, fp, labels, smiles_tokens
    X, A, mogen_fp, labels, smiles_tokens = load_data_with_smiles(path, smiles_utils)

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
    smiles_tokens = torch.LongTensor(smiles_tokens)
    labels = torch.FloatTensor(labels)

    # 6) Split train/test
    #   (A justifier dans vos vrais scripts)
    train_dataset = MyDataset(
        smiles_tokens=smiles_tokens[:8284],
        f=mogen_fp[:8284], f1=fp1[:8284], f2=fp2[:8284], f3=fp3[:8284], f4=fp4[:8284],
        X=X[:8284], A=A[:8284],
        label=labels[:8284]
    )
    test_dataset = MyDataset(
        smiles_tokens=smiles_tokens[8284:],
        f=mogen_fp[8284:], f1=fp1[8284:], f2=fp2[8284:], f3=fp3[8284:], f4=fp4[8284:],
        X=X[8284:], A=A[8284:],
        label=labels[8284:]
    )

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=True)

    # 7) Instancier le modèle
    model = MyModel(vocab_size=5000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

    # 8) Entraînement simple
    model.train()
    n_epochs = 5
    for epoch in range(n_epochs):
        pred_y = []
        pred_binary = []
        true_y = []

        for i, batch in enumerate(train_loader):
            smiles_tokens_bt, fp_bt, fp1_bt, fp2_bt, fp3_bt, fp4_bt, X_bt, A_bt, label_bt = batch

            optimizer.zero_grad()
            logits, loss = model(smiles_tokens_bt, fp_bt, fp1_bt, fp2_bt, fp3_bt, fp4_bt, X_bt, A_bt, label_bt)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Pour évaluation dans l'epoch
            logits_np = logits.detach().numpy()
            pred_y.extend(logits_np)
            pred_binary.extend(logits_np.round())
            true_y.extend(label_bt.numpy())

        if epoch == (n_epochs - 1):
            acc = accuracy_score(true_y, pred_binary)
            auc = roc_auc_score(true_y, pred_y)
            print(f"Epoch {epoch} - Loss {loss.item():.4f} - ACC {acc:.3f} - AUC {auc:.3f}")

    # 9) Évaluation sur test
    model.eval()
    pred_y_test = []
    pred_binary_test = []
    true_y_test = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            smiles_tokens_bt, fp_bt, fp1_bt, fp2_bt, fp3_bt, fp4_bt, X_bt, A_bt, label_bt = batch
            logits, loss = model(smiles_tokens_bt, fp_bt, fp1_bt, fp2_bt, fp3_bt, fp4_bt, X_bt, A_bt, label_bt)

            logits_np = logits.detach().numpy()
            pred_y_test.extend(logits_np)
            pred_binary_test.extend(logits_np.round())
            true_y_test.extend(label_bt.numpy())

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(true_y_test, pred_binary_test).ravel()
    acc_test = accuracy_score(true_y_test, pred_binary_test)
    auc_test = roc_auc_score(true_y_test, pred_y_test)

    print("Test ACC:", acc_test)
    print("Test AUC:", auc_test)

    # Autres métriques :
    spe = tn / (tn + fp)
    sen = tp / (tp + fn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    mcc = (tp*tn - fp*fn) / (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)

    print("Specificity (SPE):", spe)
    print("Sensitivity (SEN):", sen)
    print("PPV:", ppv)
    print("NPV:", npv)
    print("MCC:", mcc)
