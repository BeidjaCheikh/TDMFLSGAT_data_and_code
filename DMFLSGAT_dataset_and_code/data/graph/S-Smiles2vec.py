
import sys
sys.path.append('/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from graph.layer import GraphAttentionLayer


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, eps=0.0):
        super(GIN, self).__init__()
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=True)  # Learnable epsilon
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, inputs):
        X, A = inputs  # X: Node features, A: Adjacency matrix
        # Aggregate neighbor features
        aggregated = torch.matmul(A, X)  # Sum of neighbors' features
        # Apply GIN update
        updated = (1 + self.eps) * X + aggregated
        out = self.mlp(updated)
        return out, A


class GraphModel(nn.Module):
    def __init__(self):
        super(GraphModel, self).__init__()
        self.num_head = 4

        self.layers = nn.Sequential(
            GIN(input_dim=65, hidden_dim=65),  # Première couche GIN
            GIN(input_dim=65, hidden_dim=65),  # Deuxième couche GIN
            GIN(input_dim=65, hidden_dim=65),  # Troisième couche GIN
            GIN(input_dim=65, hidden_dim=65),  # Quatrième couche GIN
        )

        self.proj = nn.Sequential(
            nn.Linear(12800, 1024),#out_features *numheads* 100 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.att = GraphAttentionLayer()

    def forward(self, X, A):
        # GCN
        # out = self.layers((X, A))[0]

        # GAT
        features = []
        for i in range(X.shape[0]):
            feature_temp = []
            x, a = X[i], A[i]
            # 2层gat
            for _ in range(self.num_head):
                ax = self.att(x, a)
                feature_temp.append(ax)
            feature_temp = torch.cat(feature_temp, dim=1)
            features.append(feature_temp)
        out = torch.stack(features, dim=0)
        out = out.view(out.size(0), -1)
        out = self.proj(out)

        return out


class FpModel(nn.Module):
    def __init__(self):
        super(FpModel, self).__init__()

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
        self.fp2 = nn.Sequential(

            nn.Linear(166, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

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
        self.fc = nn.Linear(128, 1)

        # self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, x1, x2, x3, x4):
        '''

        '''
        x = self.fp(x)
        x1 = self.fp1(x1)
        x2 = self.fp2(x2)
        x3 = self.fp3(x3)
        x4 = self.fp4(x4)

        return x, x1, x2, x3, x4


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.graph = GraphModel()
        self.fp = FpModel()
        self.proj = nn.Sequential(
            nn.Linear(128 * 6, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, 1)
        self.active = nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, f, f1, f2, f3, f4, X, A, label):
        f, f1, f2, f3, f4 = self.fp(f, f1, f2, f3, f4)
        X = self.graph(X, A)

        x = torch.cat((f, f1, f2, f3, f4, X), dim=1)
        x = self.proj(x)
        x = self.fc(x)
        x = self.active(x).squeeze(-1)
        loss = self.loss_fn(x, label)

        return x, loss


if __name__ == '__main__':
    from DMFGAM数据集及代码.data.graph.data_utils import load_data, MyDataset, set_seed, load_fp
    from torch.utils.data import DataLoader
    import torch
    import pandas as pd

    SEED = 42
    set_seed(SEED)

    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/Smiles.csv'
    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/PubchemFP881.csv'
    MACCSFP_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/MACCSFP.csv'
    APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/APC2D780.csv'
    KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/KR.csv'
    X, A, mogen_fp, labels = load_data(path)

    fp1 = load_fp(PubchemFP881_path)
    fp2 = load_fp(MACCSFP_path)
    fp3 = load_fp(APC2D780_path)
    fp4 = load_fp(KR_path)

    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    fp1 = torch.FloatTensor(fp1)
    fp2 = torch.FloatTensor(fp2)
    fp3 = torch.FloatTensor(fp3)
    fp4 = torch.FloatTensor(fp4)
    labels = torch.FloatTensor(labels)

    # Calcul de la taille de l'entraînement et du test
    train_size = int(0.8 * len(mogen_fp))
    test_size = len(mogen_fp) - train_size

    # Division des ensembles
    train_dataset = MyDataset(f=mogen_fp[:train_size], f1=fp1[:train_size], f2=fp2[:train_size], f3=fp3[:train_size], f4=fp4[:train_size],
                            X=X[:train_size], A=A[:train_size], label=labels[:train_size])
    test_dataset = MyDataset(f=mogen_fp[train_size:], f1=fp1[train_size:],f2=fp2[train_size:],f3=fp3[train_size:], f4=fp4[train_size:],
                            X=X[train_size:], A=A[train_size:], label=labels[train_size:])

    # Chargement des données
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=True)

    print(X.shape, A.shape)
    # print(X.shape, A.shape)
    # print(X.shape, A.shape)

    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
    lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    model.train()
    for epoch in range(6):
        pred_y = []
        PED = []
        ture_y = []
        for i, batch in enumerate(train_loader):
            fp, fp1, fp2, fp3, fp4, X, A, label = batch
            optimizer.zero_grad()
            logits, loss = model(fp, fp1, fp2, fp3, fp4, X, A, label)
            # logits, loss = model(fp, label)
            loss.backward()
            optimizer.step()
            lr_optimizer.step()
            print('Epoch:', epoch, 'Loss:', loss.item())

            # logits = torch.argmax(logits, dim=1)
            logits = logits.detach().numpy()
            pred_y.extend(logits)

            PED.extend(logits.round())

            label = label.numpy()
            ture_y.extend(label)

        if epoch == 5:
            acc = accuracy_score(ture_y, PED)
            auc = roc_auc_score(ture_y, pred_y)
            print('训练集准确率ACC:', acc)
            print('训练集AUC:', auc)

    model.eval()
    pred_y = []
    PED = []
    ture_y = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            fp, fp1, fp2, fp3, fp4, X, A, label = batch

            logits, loss = model(fp, fp1, fp2, fp3, fp4, X, A, label)
            # logits, loss = model(fp, label)
            # print('Loss:', loss.item())

            # logits = torch.argmax(logits, dim=1)
            logits = logits.detach().numpy()
            pred_y.extend(logits)

            PED.extend(logits.round())
            label = label.numpy()
            ture_y.extend(label)

        TN, FP, FN, TP = confusion_matrix(ture_y, PED).ravel()
        SPE = TN / (TN + FP)
        SEN = TP / (TP + FN)
        NPV = TN / (TN + FN)
        PPV = TP / (TP + FP)
        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
        print('TN, FP, FN, TP:', TN, FP, FN, TP)
        print('SPE, SEN, NPV, PPV, MCC:', SPE, SEN, NPV, PPV, MCC)
        acc = accuracy_score(ture_y, PED)
        auc = roc_auc_score(ture_y, pred_y)
        print('测试集准确率ACC:', acc)
        print('测试集AUC:', auc)




#########################################################################################"


import sys
sys.path.append('/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from graph.layer import GraphAttentionLayer

class SmilesTransformer(nn.Module):
    """Encodeur Transformer pour les séquences SMILES."""
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=256, num_layers=2, max_len=79, dropout=0.1):
        super(SmilesTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self._generate_positional_encoding(max_len, embed_dim)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def _generate_positional_encoding(self, max_len, embed_dim):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, smiles_tokens):
        x = self.embedding(smiles_tokens) + self.positional_encoding[:, :smiles_tokens.size(1), :]
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)  # Pooling sur la séquence
        return x
    
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.weight = nn.Parameter(torch.randn(65, 65), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(65), requires_grad=True)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        X, A = inputs
        xw = torch.matmul(X, self.weight)
        out = torch.matmul(A, xw)

        out += self.bias
        out = self.relu(out)


class GraphModel(nn.Module):
    def __init__(self):
        super(GraphModel, self).__init__()
        self.num_head = 4

        self.layers = nn.Sequential(
            GCN(),
            GCN(),
            GCN(),
            GCN(),
         
        )

        self.proj = nn.Sequential(
            nn.Linear(32*4*100, 1024),#out_features *numheads* 100 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.att = GraphAttentionLayer()

    def forward(self, X, A):
        # GCN
        # out = self.layers((X, A))[0]

        # GAT
        features = []
        for i in range(X.shape[0]):
            feature_temp = []
            x, a = X[i], A[i]
            # 2层gat
            for _ in range(self.num_head):
                ax = self.att(x, a)
                feature_temp.append(ax)
            feature_temp = torch.cat(feature_temp, dim=1)
            features.append(feature_temp)
        out = torch.stack(features, dim=0)
        out = out.view(out.size(0), -1)
        out = self.proj(out)

        return out


class FpModel(nn.Module):
    def __init__(self):
        super(FpModel, self).__init__()

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
        self.fp2 = nn.Sequential(

            nn.Linear(166, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

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
        self.fc = nn.Linear(128, 1)

        # self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, x1, x2, x3, x4):
        '''

        '''
        x = self.fp(x)
        x1 = self.fp1(x1)
        x2 = self.fp2(x2)
        x3 = self.fp3(x3)
        x4 = self.fp4(x4)

        return x, x1, x2, x3, x4


class MyModel(nn.Module):
    """Full model combining SMILES Transformer, Graph Model, and Fingerprints."""
    def __init__(self, vocab_size):
        super(MyModel, self).__init__()
        self.smiles_transformer = SmilesTransformer(vocab_size)
        self.graph_model = GraphModel()
        self.fp_model = FpModel()
        self.proj = nn.Sequential(
            nn.Linear(128 * 6 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, 1)
        self.active = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, smiles_tokens, f, f1, f2, f3, f4, X, A, label):
        smiles_features = self.smiles_transformer(smiles_tokens)
        graph_features = self.graph_model(X, A)
        fp_features = self.fp_model(f, f1, f2, f3, f4)
        fp_concat = torch.cat(fp_features, dim=1)

        combined = torch.cat((smiles_features, graph_features, fp_concat), dim=1)
        x = self.proj(combined)
        x = self.fc(x)
        x = self.active(x).squeeze(-1)
        loss = self.loss_fn(x, label)
        return x, loss


if __name__ == '__main__':
    from DMFGAM数据集及代码.data.graph.data_utils import load_data_with_smiles, MyDataset, set_seed, load_fp
    from torch.utils.data import DataLoader
    from DMFGAM数据集及代码.data.graph.data_utils import SmilesUtils
    import torch
    import pandas as pd

    SEED = 42
    set_seed(SEED)

    # Initialisation du tokenizer SMILES
    smiles_utils = SmilesUtils(vocab_size=5000, max_len=79)
    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/Smiles.csv'

    # Charger les données et entraîner le tokenizer SMILES
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()
    smiles_utils.train_tokenizer(smiles_list)

    # Charger les données avec SMILES tokenisés
    X, A, mogen_fp, labels, smiles_tokens = load_data_with_smiles(path, smiles_utils)

    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/PubchemFP881.csv'
    MACCSFP_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/MACCSFP.csv'
    APC2D780_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/APC2D780.csv'
    KR_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code/DMFGAM数据集及代码/data/FP/KR.csv'

    # Charger les empreintes moléculaires
    fp1 = torch.FloatTensor(load_fp(PubchemFP881_path))
    fp2 = torch.FloatTensor(load_fp(MACCSFP_path))
    fp3 = torch.FloatTensor(load_fp(APC2D780_path))
    fp4 = torch.FloatTensor(load_fp(KR_path))

    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    smiles_tokens = torch.LongTensor(smiles_tokens)
    labels = torch.FloatTensor(labels)

    # Calcul de la taille de l'entraînement et du test
    train_size = int(0.8 * len(mogen_fp))
    test_size = len(mogen_fp) - train_size

    # Division des ensembles
    train_dataset = MyDataset(
        smiles_tokens=smiles_tokens[:train_size],
        f=mogen_fp[:train_size], f1=fp1[:train_size], f2=fp2[:train_size], f3=fp3[:train_size], f4=fp4[:train_size],
        X=X[:train_size], A=A[:train_size], label=labels[:train_size]
    )
    test_dataset = MyDataset(
        smiles_tokens=smiles_tokens[train_size:],
        f=mogen_fp[train_size:], f1=fp1[train_size:], f2=fp2[train_size:], f3=fp3[train_size:], f4=fp4[train_size:],
        X=X[train_size:], A=A[train_size:], label=labels[train_size:]
    )

    # Chargement des données
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=True)

    # Initialisation du modèle
    model = MyModel(vocab_size=5000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    # Entraînement du modèle
    model.train()
    for epoch in range(5):
        pred_y = []
        PED = []
        ture_y = []
        for i, batch in enumerate(train_loader):
            smiles_tokens, fp, fp1, fp2, fp3, fp4, X, A, label = batch
            optimizer.zero_grad()
            logits, loss = model(smiles_tokens, fp, fp1, fp2, fp3, fp4, X, A, label)
            loss.backward()
            optimizer.step()
            lr_optimizer.step()
            print('Epoch:', epoch, 'Loss:', loss.item())

            logits = logits.detach().numpy()
            pred_y.extend(logits)
            PED.extend(logits.round())
            ture_y.extend(label.numpy())

        if epoch == 4:
            acc = accuracy_score(ture_y, PED)
            auc = roc_auc_score(ture_y, pred_y)
            print('Training Accuracy ACC:', acc)
            print('Training AUC:', auc)

    # Évaluation du modèle
    model.eval()
    pred_y = []
    PED = []
    ture_y = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            smiles_tokens, fp, fp1, fp2, fp3, fp4, X, A, label = batch
            logits, loss = model(smiles_tokens, fp, fp1, fp2, fp3, fp4, X, A, label)

            logits = logits.detach().numpy()
            pred_y.extend(logits)
            PED.extend(logits.round())
            ture_y.extend(label.numpy())

        TN, FP, FN, TP = confusion_matrix(ture_y, PED).ravel()
        SPE = TN / (TN + FP)
        SEN = TP / (TP + FN)
        NPV = TN / (TN + FN)
        PPV = TP / (TP + FP)
        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
        acc = accuracy_score(ture_y, PED)
        auc = roc_auc_score(ture_y, pred_y)

        print('TN, FP, FN, TP:', TN, FP, FN, TP)
        print('SPE, SEN, NPV, PPV, MCC:', SPE, SEN, NPV, PPV, MCC)
        print('Test Accuracy ACC:', acc)
        print('Test AUC:', auc)




















