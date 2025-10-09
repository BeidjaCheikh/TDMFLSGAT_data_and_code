import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import torch
from torch.utils.data import DataLoader
from DMFGAM数据集及代码.data.graph.DMFLSGAT import MyModel  # Importez votre modèle principal
from DMFGAM数据集及代码.data.graph.data_utils import load_data, MyDataset, set_seed, load_fp
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Définir l'espace de recherche pour les hyperparamètres
space = {
    'lr': hp.loguniform('lr', -5, -3),  # Learning rate entre 10^-5 et 10^-3
    'dropout': hp.uniform('dropout', 0.0, 0.5),  # Dropout entre 0.0 et 0.5
    'weight_decay': hp.loguniform('weight_decay', -6, -2),  # Regularization
    'batch_size': hp.choice('batch_size', [64, 128, 256]),  # Batch size
}

# Chargement des données (définissez les chemins appropriés pour vos fichiers)
def load_dataset():
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

    train_size = int(0.8 * len(mogen_fp))
    test_size = len(mogen_fp) - train_size

    train_dataset = MyDataset(
        f=mogen_fp[:train_size],
        f1=fp1[:train_size],
        f2=fp2[:train_size],
        f3=fp3[:train_size],
        f4=fp4[:train_size],
        X=X[:train_size],
        A=A[:train_size],
        label=labels[:train_size],
    )
    test_dataset = MyDataset(
        f=mogen_fp[train_size:],
        f1=fp1[train_size:],
        f2=fp2[train_size:],
        f3=fp3[train_size:],
        f4=fp4[train_size:],
        X=X[train_size:],
        A=A[train_size:],
        label=labels[train_size:],
    )
    return train_dataset, test_dataset

# Fonction pour évaluer un ensemble d'hyperparamètres
def objective(params):
    print(f"Essai avec les paramètres : {params}")
    train_dataset, test_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=int(params['batch_size']), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(params['batch_size']), shuffle=False)

    # Instanciation du modèle
    model = MyModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    model.train()

    # Entraînement
    for epoch in range(5):
        for i, batch in enumerate(train_loader):
            fp, fp1, fp2, fp3, fp4, X, A, label = batch
            optimizer.zero_grad()
            logits, loss = model(fp, fp1, fp2, fp3, fp4, X, A, label)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    # Évaluation
    model.eval()
    pred_y = []
    true_y = []
    with torch.no_grad():
        for batch in test_loader:
            fp, fp1, fp2, fp3, fp4, X, A, label = batch
            logits, _ = model(fp, fp1, fp2, fp3, fp4, X, A, label)
            pred_y.extend(logits.numpy())
            true_y.extend(label.numpy())

    # Calcul des métriques
    auc = roc_auc_score(true_y, pred_y)
    print(f"AUC obtenu : {auc}")
    return {'loss': -auc, 'status': STATUS_OK}

# Recherche d'hyperparamètres
def hyperparameter_search():
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,  # Nombre maximum d'itérations
        trials=trials,
    )
    print(f"Meilleurs hyperparamètres : {best}")
    return best

if __name__ == '__main__':
    set_seed(42)  # Assurez-vous d'avoir la même fonction `set_seed`
    best_params = hyperparameter_search()
    print(f"Recherche terminée. Meilleurs paramètres : {best_params}")
