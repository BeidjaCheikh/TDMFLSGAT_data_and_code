##model.py
import sys
sys.path.append('/home/enset/Téléchargements/DMFLSGAT_data_and_code/data')
import torch
import torch.nn as nn
import torch.nn.functional as F
from DMFLSGAT_dataset_and_code.data.graph.LSGATLayer import LSGATLayer

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

        
##############################################################
#                2. Deep LSGAT GraphModel                    #
##############################################################
class GraphModel(nn.Module):
    def __init__(self, num_layers=5, num_heads=4, in_features=65, out_features=32, dropout=0.5, alpha=0.2, beta=0.6, max_nodes=100):
        super(GraphModel, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_features = out_features
        self.max_nodes = max_nodes

        self.lsgat_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_features if i == 0 else out_features * num_heads
            self.lsgat_layers.append(
                nn.ModuleList([
                    LSGATLayer(
                        in_features=input_dim,
                        out_features=out_features,
                        dropout=dropout,
                        alpha=alpha,
                        concat=True,
                        beta=beta,
                        layer_id=i+1
                    )
                    for _ in range(num_heads)
                ])
            )
        # Projection finale
        self.proj = nn.Sequential(
            nn.Linear(out_features * num_heads * max_nodes, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, X, A):
        # X: (batch, N, in_features), A: (batch, N, N)
        batch_size, N, _ = X.shape
        x = X
        for layer in self.lsgat_layers:
            head_outputs = []
            for lsgat in layer:
                temp = []
                for i in range(batch_size):
                    temp.append(lsgat(x[i], A[i]))
                temp = torch.stack(temp, dim=0)  # (batch, N, out_features)
                head_outputs.append(temp)
            x = torch.cat(head_outputs, dim=2)  # (batch, N, out_features * num_heads)
        out = x.view(batch_size, -1)  # Flatten pour la couche fully connected
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
    def __init__(self, vocab_size):
        super(MyModel, self).__init__()
        self.smiles_transformer = SmilesTransformer(vocab_size)
        self.graph_model = GraphModel()
        self.fp_model = FpModel()
        self.proj = nn.Sequential(
            nn.Linear(128 * 7, 128),  # 1 (smiles) + 1 (graph) + 5 (fp)
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