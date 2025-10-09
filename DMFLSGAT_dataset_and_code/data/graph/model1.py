import sys
sys.path.append('/home/enset/Téléchargements/DMFLSGAT_data_and_code/data')
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph.layer1 import GraphAttentionLayer

##############################################################
#                1. SMILES Transformer                       #
##############################################################
class SmilesTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=256, num_layers=6, max_len=79, dropout=0.1):
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
        pe = self.positional_encoding.to(smiles_tokens.device)
        x = self.embedding(smiles_tokens) + pe[:, :smiles_tokens.size(1), :]
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)  
        return x
    
class GCN(nn.Module):
    """Exemple de couche GCN simple."""
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
        return out, A
    
class GraphModel(nn.Module):
    """Exemple d’utilisation de GAT."""
    def __init__(self):
        super(GraphModel, self).__init__()
        self.num_head = 4
        self.proj = nn.Sequential(
            nn.Linear(32*4*100, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.att = GraphAttentionLayer()

    def forward(self, X, A):
        features = []
        for i in range(X.shape[0]):
            feature_temp = []
            x, a = X[i], A[i]
            for _ in range(self.num_head):
                ax = self.att(x, a)
                feature_temp.append(ax)
            feature_temp = torch.cat(feature_temp, dim=1)
            features.append(feature_temp)
        out = torch.stack(features, dim=0)
        out = out.view(out.size(0), -1)
        out = self.proj(out)
        return out


##############################################################
#                  3. Fingerprint Model                      #
##############################################################
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
        # self.fc = nn.Linear(128, 1)  # Non utilisé directement

    def forward(self, x, x1, x2, x3, x4):
        x = self.fp(x)
        x1 = self.fp1(x1)
        x2 = self.fp2(x2)
        x3 = self.fp3(x3)
        x4 = self.fp4(x4)
        return x, x1, x2, x3, x4

##############################################################
#                  4. Modèle Global                          #
##############################################################
class MyModel(nn.Module):
    """Modèle complet combinant SMILES Transformer, Graph Model, et Fingerprints."""
    def __init__(self, vocab_size):
        super(MyModel, self).__init__()
        self.smiles_transformer = SmilesTransformer(vocab_size)
        self.graph_model = GraphModel()
        self.fp_model = FpModel()
        self.proj = nn.Sequential(
            nn.Linear(128 * 7, 128),  # 7 = (1 sortie smiles + 1 sortie graph + 5 sorties fp)
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

