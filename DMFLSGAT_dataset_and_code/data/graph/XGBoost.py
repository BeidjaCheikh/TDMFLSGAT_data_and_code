import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score
)
from xgboost import XGBClassifier

if __name__ == '__main__':
    from Utils import load_data, set_seed, load_fp

    SEED = 42
    set_seed(SEED)

    path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/DMFGAM数据集及代码/data/graph/Dataset/Internal dataset/Smiles.csv'
    PubchemFP881_path = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/PubchemFP881.csv'
    GraphFP1024_path  = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/GraphFP1024.csv'
    APC2D780_path     = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/APC2D780.csv'
    FP1024_path       = r'/home/enset/Téléchargements/DMFGAM_data_and_code12/FP1024.csv'

    X, A, mogen_fp, labels = load_data(path)
    fp1 = load_fp(PubchemFP881_path)
    fp2 = load_fp(GraphFP1024_path)
    fp3 = load_fp(APC2D780_path)
    fp4 = load_fp(FP1024_path)

    mogen_fp = torch.FloatTensor(mogen_fp)
    fp1 = torch.FloatTensor(fp1)
    fp2 = torch.FloatTensor(fp2)
    fp3 = torch.FloatTensor(fp3)
    fp4 = torch.FloatTensor(fp4)
    labels = torch.FloatTensor(labels)

    # ============================================================
    # Concat toutes les features (mogen_fp + fp1 + fp2 + fp3 + fp4)
    # ============================================================
    feat = torch.cat([mogen_fp, fp1, fp2, fp3, fp4], dim=1)

    X_train = feat[0:8000].numpy()
    y_train = labels[0:8000].numpy().astype(int)

    X_test  = feat[8000:10355].numpy()
    y_test  = labels[8000:10355].numpy().astype(int)

    # =============
    # XGBoost model
    # =============
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    TN, FP, FN, TP = confusion_matrix(y_test, pred).ravel()
    SPE = TN / (TN + FP)
    SEN = TP / (TP + FN)
    NPV = TN / (TN + FN)
    PPV = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

    ACC = accuracy_score(y_test, pred)
    AUC = roc_auc_score(y_test, proba)

    AP = average_precision_score(y_test, proba)

    print("=== XGBoost ===")
    print("TN, FP, FN, TP:", TN, FP, FN, TP)
    print(
        "SPE, SEN, NPV, PPV, MCC:",
        f"{SPE:.3f}",
        f"{SEN:.3f}",
        f"{NPV:.3f}",
        f"{PPV:.3f}",
        f"{MCC:.3f}"
    
    )
    print("Test set accuracy (ACC):", f"{ACC:.3f}")
    print("Test set AUC:", f"{AUC:.3f}")
    print("AP (Average Precision / AUPRC):", f"{AP:.3f}")