import torch
import torch.nn as nn
import torch.nn.functional as F
from DMFLSGAT_dataset_and_code.data.graph.LSGATLayer import GraphAttentionLayer
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


if __name__ == '__main__':
    from DMFLSGAT_dataset_and_code.data.graph.DMFLSGATUtils import load_data, MyDataset, set_seed, load_fp
    from torch.utils.data import DataLoader
    import torch
    import pandas as pd

    SEED = 42
    set_seed(SEED)

    path = r'C:\Users\DELL\Downloads\DMFGAM_data_and_code12\DMFLSGAT_dataset_and_code\data\Smiles.csv'
    PubchemFP881_path = r'C:\Users\DELL\Downloads\DMFGAM_data_and_code12\DMFLSGAT_dataset_and_code\data\Pubchem881.csv'
    GraphFP1024_path = r'C:\Users\DELL\Downloads\DMFGAM_data_and_code12\DMFLSGAT_dataset_and_code\data\GraphFP1024.csv'
    APC2D780_path = r'C:\Users\DELL\Downloads\DMFGAM_data_and_code12\DMFLSGAT_dataset_and_code\data\APC2D780.csv'
    FP1024_path = r'C:\Users\DELL\Downloads\DMFGAM_data_and_code12\DMFLSGAT_dataset_and_code\data\FP1024.csv'
    X, A, mogen_fp, labels = load_data(path) 

    fp1 = load_fp(PubchemFP881_path)
    fp2 = load_fp(GraphFP1024_path)
    fp3 = load_fp(APC2D780_path)
    fp4 = load_fp(FP1024_path)

    X = torch.FloatTensor(X)
    A = torch.FloatTensor(A)
    mogen_fp = torch.FloatTensor(mogen_fp)
    fp1 = torch.FloatTensor(fp1)
    fp2 = torch.FloatTensor(fp2)
    fp3 = torch.FloatTensor(fp3)
    fp4 = torch.FloatTensor(fp4)
    labels = torch.FloatTensor(labels)

    from sklearn.ensemble import RandomForestClassifier

    # 构建模型，用决策树和随机森林进行对比
    model = RandomForestClassifier(random_state=0)

    model.fit(mogen_fp[0:8000], labels[0:8000])

    # score_r = model.score(mogen_fp[8000:10355], labels[8000:10355])

    pred = model.predict(mogen_fp[8000:10355])

    TN, FP, FN, TP = confusion_matrix(labels[8000:10355], pred).ravel()
    SPE = TN / (TN + FP)
    SEN = TP / (TP + FN)
    NPV = TN / (TN + FN)
    PPV = TP / (TP + FP)

    MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    print('TN, FP, FN, TP:', TN, FP, FN, TP)
    print('SPE, SEN, NPV, PPV, MCC:', SPE, SEN, NPV, PPV, MCC)
    acc = accuracy_score(labels[8000:10355], pred)
    auc = roc_auc_score(labels[8000:10355], pred)
    print('Test set accuracy (ACC):', acc)
    print('Test set AUC:', auc)

