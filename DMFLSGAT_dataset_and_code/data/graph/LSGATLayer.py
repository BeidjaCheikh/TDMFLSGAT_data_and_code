##layer
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSGATLayer(nn.Module):
    """
    LSGAT : Layer-wise Self-Adaptive Graph Attention Network (Su et al., 2024)
    https://doi.org/10.1016/j.knosys.2024.111649
    """
    def __init__(self, in_features=65, out_features=32, dropout=0.5, alpha=0.2, concat=True, beta=0.6, layer_id=1):
        super(LSGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.beta = beta      # Hyperparamètre, proportion de nœuds à "regularizer" (voir Eq.6-8)
        self.layer_id = layer_id  # Pour moduler l'effet selon la profondeur

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh) # (N, N)

        # == Calcul overlap-degree et scaling score selon Su et al. (Eq. 6-8) ==
        # 1. Overlap-degree d(j) = (A^layer_id @ 1)[j]
        N = adj.shape[0]
        device = adj.device
        A_power = torch.matrix_power(adj + torch.eye(N, device=device), self.layer_id)
        overlap_degree = A_power.sum(dim=0)   # (N,) vecteur, chaque noeud j

        # 2. Calcul du seuil tau, puis du scaling score (Eq. 7-8)
        tau = torch.quantile(overlap_degree, self.beta)
        scaled_overlap = overlap_degree / (tau + 1e-8)
        scaling_score = torch.where(scaled_overlap <= 1, torch.ones_like(scaled_overlap), 1 / scaled_overlap) # (N,)

        # 3. Injection du scaling dans le mécanisme d'attention avant LeakyReLU+softmax
        #    e[i,j] * scaling_score[j]
        e = e * scaling_score.unsqueeze(0) # (N, N)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(self.leakyrelu(attention), dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T # (N, N)
        return e

    def __repr__(self):
        return self.__class__.__name__ + f' ({self.in_features} -> {self.out_features})'