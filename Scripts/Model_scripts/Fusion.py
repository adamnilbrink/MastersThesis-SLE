import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas_plink import read_plink
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"


##### 1) CONFIGURATION & PATHS ETC #####

BASE_DIR = "/Users/adani406/SLE"

PLINK_PREFIX = os.path.join(BASE_DIR, "G1_S1", "set1mainvars")
GENETIC_SNP_FILE = os.path.join(BASE_DIR, "Final_SNP_Selection.xlsx")
CLINICAL_FILE = os.path.join(BASE_DIR, "SLE_INTEGRATED_COHORT_COMBINED.xlsx")

TARGET_SHEET_CANDIDATES = ["Consensus SNPs"]
CLINICAL_SHEET_CANDIDATES = ["Test-Train_Final"]

IID_COL = "IID"
LN_COL = "LN"
TRAIN_COL = "Train"
TEST_COL = "Test"
FOLLOW_UP_COL = "<6y follow-up"

CLINICAL_COLS = [
    "Age of diagnosis",
    "Gender",
    "SSA",
    "Sm",
    "RNP",
    "aCL_IgM",
    "aCL_IgG",
    "Anti-dsDNA",
]

RANDOM_STATE = 42


#    Same colors used across all plots for consistency and aesthetics
PLOT_BLUE = "#1a55f5"
PLOT_GREEN = "#0e9152"
PLOT_ORANGE = "#f5841a"
PLOT_GRAY = "#7a7a7a"
FIGSIZE_LARGE = (6.8, 5.1)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# Konservativ OOF-selection för att minska överanpassning
OOF_TOLERANCE = 0.003

# Genetisk RF används endast för att skapa GRS

BEST_GENETIC_RF_PARAMS = {
    "n_estimators": 1800,
    "max_depth": 8,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "criterion": "gini",
    "class_weight": "balanced_subsample",
    "max_samples": None,
}

GENETIC_RF_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 123]
GENETIC_CV_SPLITS = 5
GENETIC_CV_REPEATS = 5

KNN_NEIGHBORS = 5

OUTER_CV_SPLITS = 5
OUTER_CV_REPEATS = 5

INNER_CV_SPLITS = 5
INNER_CV_REPEATS = 3

# Lättare inner CV för tree-modeller för att minska flaskhals
TREE_INNER_CV_SPLITS = 5
TREE_INNER_CV_REPEATS = 1

ENET_C_GRID = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
ENET_L1_RATIO_GRID = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

# Gemensamma viktningar för fusionmodellerna
STATIC_WEIGHT_GRID = np.round(np.arange(0.60, 0.951, 0.025), 3).tolist()

# Gridar för klinisk RF och XGB
CLINICAL_RF_TUNING_SEEDS = [11, 22]
CLINICAL_RF_FINAL_SEEDS = [11, 22, 33]

CLINICAL_XGB_TUNING_SEEDS = [11, 22]
CLINICAL_XGB_FINAL_SEEDS = [11, 22, 33]

CLINICAL_RF_PARAM_GRID = [
    {
        "n_estimators": 600,
        "max_depth": 3,
        "min_samples_split": 14,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "criterion": "gini",
        "class_weight": "balanced_subsample",
        "max_samples": 0.75,
    },
    {
        "n_estimators": 900,
        "max_depth": 4,
        "min_samples_split": 12,
        "min_samples_leaf": 6,
        "max_features": "sqrt",
        "criterion": "gini",
        "class_weight": "balanced_subsample",
        "max_samples": 0.80,
    },
    {
        "n_estimators": 1200,
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "criterion": "gini",
        "class_weight": "balanced_subsample",
        "max_samples": 0.85,
    },
    {
        "n_estimators": 1200,
        "max_depth": 5,
        "min_samples_split": 12,
        "min_samples_leaf": 6,
        "max_features": 0.50,
        "criterion": "gini",
        "class_weight": "balanced_subsample",
        "max_samples": 0.80,
    },
]

CLINICAL_XGB_PARAM_GRID = [
    {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "gamma": 0.4,
    },
    {
        "n_estimators": 500,
        "max_depth": 2,
        "learning_rate": 0.02,
        "min_child_weight": 12,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.5,
        "reg_lambda": 6.0,
        "gamma": 0.5,
    },
    {
        "n_estimators": 700,
        "max_depth": 2,
        "learning_rate": 0.015,
        "min_child_weight": 14,
        "subsample": 0.65,
        "colsample_bytree": 0.60,
        "reg_alpha": 2.0,
        "reg_lambda": 7.0,
        "gamma": 0.6,
    },
    {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.02,
        "min_child_weight": 12,
        "subsample": 0.60,
        "colsample_bytree": 0.60,
        "reg_alpha": 2.0,
        "reg_lambda": 8.0,
        "gamma": 0.7,
    },
]


##### 2. HJÄLPFUNKTIONER #####

def clean_id(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.replace("SLE-LIU1_", "")


def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce")


def pick_sheet_name(xlsx_path, candidates):
    xls = pd.ExcelFile(xlsx_path)
    for name in candidates:
        if name in xls.sheet_names:
            return name
    raise ValueError(f"Hittade inget matchande ark i {xlsx_path}")


def find_snp_column(df, sheet_name):
    for col in ["SNP", "SNV"]:
        if col in df.columns:
            return col
    raise ValueError(f"Arket '{sheet_name}' saknar kolumnen 'SNP' eller 'SNV'.")


def load_snp_names(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    snp_col = find_snp_column(df, sheet_name)
    snps = df[snp_col].astype(str).str.strip().replace("nan", np.nan).dropna()
    snps = snps[snps != ""].drop_duplicates()
    return snps.tolist()


def conservative_top_row(df, score_cols, tolerance=0.003, prefer_cols=None, ascending=None):
    work = df.copy()
    best_values = work[score_cols].max()
    mask = np.ones(len(work), dtype=bool)

    for col in score_cols:
        mask &= work[col] >= (best_values[col] - tolerance)

    candidate_df = work.loc[mask].copy()
    if candidate_df.empty:
        candidate_df = work.copy()

    sort_cols = list(score_cols)
    sort_ascending = [False] * len(score_cols)

    if prefer_cols is not None and ascending is not None:
        sort_cols += list(prefer_cols)
        sort_ascending += list(ascending)

    candidate_df = candidate_df.sort_values(sort_cols, ascending=sort_ascending).reset_index(drop=True)
    return candidate_df.iloc[0]


def choose_final_fusion_candidate(candidate_df, tolerance=0.003):
    priority_map = {
        "Clinical only": 0,
        "Static blend": 1,
    }
    work = candidate_df.copy()

    best_auc = work["OOF_AUROC"].max()
    best_auprc = work["OOF_AUPRC"].max()

    mask = (
        (work["OOF_AUROC"] >= best_auc - tolerance)
        & (work["OOF_AUPRC"] >= best_auprc - tolerance)
    )
    cand = work.loc[mask].copy()
    if cand.empty:
        cand = work.copy()

    cand["Model_Priority"] = cand["Model"].map(priority_map)
    cand = cand.sort_values(
        ["OOF_AUROC", "OOF_AUPRC", "Model_Priority"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return cand.iloc[0]


def get_selected_oof_row(candidate_oof_df, final_model_name):
    row = candidate_oof_df.loc[candidate_oof_df["Model"] == final_model_name].copy()
    if row.empty:
        raise ValueError(f"Hittade ingen OOF-rad för vald modell: {final_model_name}")
    return row.iloc[0]


def subset_auc_auprc(y_true, probs):
    return roc_auc_score(y_true, probs), average_precision_score(y_true, probs)


##### 3. MODELLFUNKTIONER #####


def make_rf_model(params, seed):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        criterion=params["criterion"],
        class_weight=params["class_weight"],
        max_samples=params["max_samples"],
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )


def make_xgb_model(params, seed, scale_pos_weight):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        gamma=params["gamma"],
    )


##### 4. FEATURE ENGINEERING OCH TRAIN-FITTAD KLINISK PREPROCESSING #####


def build_genetic_features(genotype_matrix, snp_names):
    genotype_matrix = np.asarray(genotype_matrix, dtype=float)
    base = pd.DataFrame(genotype_matrix, columns=snp_names)

    dominant = (base >= 1).astype(float)
    dominant.columns = [f"{col}_DOM" for col in base.columns]

    recessive = (base == 2).astype(float)
    recessive.columns = [f"{col}_REC" for col in base.columns]

    features = pd.concat([base, dominant, recessive], axis=1)
    features["BURDEN_SUM"] = base.sum(axis=1)
    features["N_NONZERO"] = (base > 0).sum(axis=1).astype(float)
    features["N_HET"] = (base == 1).sum(axis=1).astype(float)
    features["N_HOMALT"] = (base == 2).sum(axis=1).astype(float)
    features["MEAN_GENO"] = base.mean(axis=1)
    features["STD_GENO"] = base.std(axis=1).fillna(0.0)
    features["MAX_GENO"] = base.max(axis=1)
    features["MIN_GENO"] = base.min(axis=1)

    return features


def split_binary_and_continuous(train_df, clinical_cols):
    binary_cols = []
    continuous_cols = []

    for col in clinical_cols:
        values = pd.to_numeric(train_df[col], errors="coerce").dropna().unique()
        value_set = set(values.tolist())

        if len(values) > 0 and value_set.issubset({0, 1}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    return binary_cols, continuous_cols


def fit_clinical_preprocessor(train_df, clinical_cols, n_neighbors=5):
    """
    Fit:as ENDAST på trainingdelen.
    Samma imputer, scaler, binary/continuous-indelning och kolumnordning återanvänds sedan
    för validation/test/external data.
    """
    clin = train_df[clinical_cols].copy()
    for col in clinical_cols:
        clin[col] = pd.to_numeric(clin[col], errors="coerce")

    binary_cols, continuous_cols = split_binary_and_continuous(clin, clinical_cols)

    n_neighbors_eff = min(n_neighbors, max(len(clin), 1))
    imputer = KNNImputer(n_neighbors=n_neighbors_eff)
    imputed_array = imputer.fit_transform(clin)
    imputed = pd.DataFrame(imputed_array, columns=clinical_cols, index=clin.index)

    for col in binary_cols:
        imputed[col] = (imputed[col] >= 0.5).astype(int)

    scaler = None
    if continuous_cols:
        scaler = StandardScaler()
        scaler.fit(imputed[continuous_cols])

    output_columns = [f"{col}_scaled" for col in continuous_cols] + binary_cols

    return {
        "clinical_cols": list(clinical_cols),
        "binary_cols": binary_cols,
        "continuous_cols": continuous_cols,
        "imputer": imputer,
        "scaler": scaler,
        "output_columns": output_columns,
    }


def transform_clinical_with_preprocessor(df, preprocessor):
    clinical_cols = preprocessor["clinical_cols"]
    binary_cols = preprocessor["binary_cols"]
    continuous_cols = preprocessor["continuous_cols"]
    imputer = preprocessor["imputer"]
    scaler = preprocessor["scaler"]
    output_columns = preprocessor["output_columns"]

    clin = df[clinical_cols].copy()
    for col in clinical_cols:
        clin[col] = pd.to_numeric(clin[col], errors="coerce")

    imputed = pd.DataFrame(
        imputer.transform(clin),
        columns=clinical_cols,
        index=clin.index,
    )

    for col in binary_cols:
        imputed[col] = (imputed[col] >= 0.5).astype(int)

    out = pd.DataFrame(index=imputed.index)

    if continuous_cols:
        scaled = scaler.transform(imputed[continuous_cols])
        for i, col in enumerate(continuous_cols):
            out[f"{col}_scaled"] = scaled[:, i]

    for col in binary_cols:
        out[col] = imputed[col].values

    return out[output_columns].reset_index(drop=True)


def preprocess_clinical_train_test(train_df, test_df, clinical_cols, n_neighbors=5):
    preprocessor = fit_clinical_preprocessor(train_df, clinical_cols, n_neighbors=n_neighbors)
    X_train = transform_clinical_with_preprocessor(train_df, preprocessor)
    X_test = transform_clinical_with_preprocessor(test_df, preprocessor)
    return X_train, X_test, preprocessor


##### 5. GENETISK PRS (GRS) #####

def get_rf_ensemble_probs(X_train, y_train, X_pred, params, seeds):
    probs_all = []
    for seed in seeds:
        model = make_rf_model(params, seed)
        model.fit(X_train, y_train)
        probs_all.append(model.predict_proba(X_pred)[:, 1])
    return np.mean(np.vstack(probs_all), axis=0)


def build_oof_rf_score(X_train, y_train, params, seeds):
    cv = RepeatedStratifiedKFold(
        n_splits=GENETIC_CV_SPLITS,
        n_repeats=GENETIC_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    oof_matrix = np.full((len(y_train), GENETIC_CV_SPLITS * GENETIC_CV_REPEATS), np.nan)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        fold_probs = get_rf_ensemble_probs(
            X_train=X_train.iloc[tr_idx],
            y_train=y_train[tr_idx],
            X_pred=X_train.iloc[va_idx],
            params=params,
            seeds=seeds,
        )
        oof_matrix[va_idx, fold_id] = fold_probs

    return np.nanmean(oof_matrix, axis=1)


##### 6. KLINISK LR #####

def score_elastic_net_cv(X_train, y_train, C, l1_ratio):
    cv = RepeatedStratifiedKFold(
        n_splits=INNER_CV_SPLITS,
        n_repeats=INNER_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    aucs = []
    auprcs = []

    for tr_idx, va_idx in cv.split(X_train, y_train):
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=10000,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train.iloc[tr_idx], y_train[tr_idx])

        probs = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_train[va_idx], probs))
        auprcs.append(average_precision_score(y_train[va_idx], probs))

    return float(np.mean(aucs)), float(np.mean(auprcs))


def tune_clinical_model(X_train, y_train):
    rows = []

    for C in ENET_C_GRID:
        for l1_ratio in ENET_L1_RATIO_GRID:
            mean_auc, mean_auprc = score_elastic_net_cv(X_train, y_train, C, l1_ratio)
            rows.append({
                "C": C,
                "l1_ratio": l1_ratio,
                "cv_auc": mean_auc,
                "cv_auprc": mean_auprc,
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["cv_auc", "cv_auprc", "C"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def _direction_from_coef(value):
    if pd.isna(value):
        return "Higher value -> higher LN probability"
    if value > 0:
        return "Higher value -> higher LN probability"
    if value < 0:
        return "Higher value -> lower LN probability"
    return "No directional effect"


def fit_clinical_model(X_train, y_train, X_test):
    search_df = tune_clinical_model(X_train, y_train)
    best_row = conservative_top_row(
        search_df,
        score_cols=["cv_auc", "cv_auprc"],
        tolerance=OOF_TOLERANCE,
        prefer_cols=["C", "l1_ratio"],
        ascending=[True, True],
    )

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=best_row["C"],
        l1_ratio=best_row["l1_ratio"],
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]

    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_[0],
    })
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df["Direction"] = coef_df["Coefficient"].map(_direction_from_coef)
    coef_df["Component_Type"] = "Clinical coefficient"
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "search_df": search_df,
        "coef_df": coef_df,
        "test_probs": test_probs,
        "best_C": best_row["C"],
        "best_l1_ratio": best_row["l1_ratio"],
    }


##### 7. KLINISK RF / XGB #####

def score_clinical_rf_cv(X_train, y_train, params):
    cv = RepeatedStratifiedKFold(
        n_splits=TREE_INNER_CV_SPLITS,
        n_repeats=TREE_INNER_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    aucs = []
    auprcs = []

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        preds = []
        for seed in CLINICAL_RF_TUNING_SEEDS:
            model = make_rf_model(params, seed + fold_id)
            model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            preds.append(model.predict_proba(X_train.iloc[va_idx])[:, 1])

        probs = np.mean(np.vstack(preds), axis=0)
        aucs.append(roc_auc_score(y_train[va_idx], probs))
        auprcs.append(average_precision_score(y_train[va_idx], probs))

    return float(np.mean(aucs)), float(np.mean(auprcs))


def tune_clinical_rf_model(X_train, y_train):
    rows = []
    for params in CLINICAL_RF_PARAM_GRID:
        mean_auc, mean_auprc = score_clinical_rf_cv(X_train, y_train, params)
        rows.append({
            "params_json": json.dumps(params, sort_keys=True),
            "params": params,
            "cv_auc": mean_auc,
            "cv_auprc": mean_auprc,
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["cv_auc", "cv_auprc"], ascending=[False, False])
        .reset_index(drop=True)
    )


def fit_clinical_rf_model(X_train, y_train, X_test):
    search_df = tune_clinical_rf_model(X_train, y_train)
    best_row = conservative_top_row(
        search_df,
        score_cols=["cv_auc", "cv_auprc"],
        tolerance=OOF_TOLERANCE,
        prefer_cols=["max_depth", "min_samples_leaf"],
        ascending=[True, False],
    )
    best_params = best_row["params"]

    preds = []
    importances = []
    for seed in CLINICAL_RF_FINAL_SEEDS:
        model = make_rf_model(best_params, seed)
        model.fit(X_train, y_train)
        preds.append(model.predict_proba(X_test)[:, 1])
        importances.append(model.feature_importances_)

    test_probs = np.mean(np.vstack(preds), axis=0)

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Mean_Importance": np.mean(np.vstack(importances), axis=0),
        "SD_Importance": np.std(np.vstack(importances), axis=0),
    }).sort_values("Mean_Importance", ascending=False).reset_index(drop=True)

    return {
        "search_df": search_df.drop(columns=["params"], errors="ignore"),
        "best_params": best_params,
        "importance_df": importance_df,
        "test_probs": test_probs,
        "best_params_json": best_row["params_json"],
    }


def score_clinical_xgb_cv(X_train, y_train, params):
    cv = RepeatedStratifiedKFold(
        n_splits=TREE_INNER_CV_SPLITS,
        n_repeats=TREE_INNER_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    aucs = []
    auprcs = []

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        y_tr = y_train[tr_idx]
        n_pos = np.sum(y_tr == 1)
        n_neg = np.sum(y_tr == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)

        preds = []
        for seed in CLINICAL_XGB_TUNING_SEEDS:
            model = make_xgb_model(params, seed + fold_id, scale_pos_weight)
            model.fit(X_train.iloc[tr_idx], y_tr)
            preds.append(model.predict_proba(X_train.iloc[va_idx])[:, 1])

        probs = np.mean(np.vstack(preds), axis=0)
        aucs.append(roc_auc_score(y_train[va_idx], probs))
        auprcs.append(average_precision_score(y_train[va_idx], probs))

    return float(np.mean(aucs)), float(np.mean(auprcs))


def tune_clinical_xgb_model(X_train, y_train):
    rows = []
    for params in CLINICAL_XGB_PARAM_GRID:
        mean_auc, mean_auprc = score_clinical_xgb_cv(X_train, y_train, params)
        rows.append({
            "params_json": json.dumps(params, sort_keys=True),
            "params": params,
            "cv_auc": mean_auc,
            "cv_auprc": mean_auprc,
            "max_depth": params["max_depth"],
            "n_estimators": params["n_estimators"],
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["cv_auc", "cv_auprc"], ascending=[False, False])
        .reset_index(drop=True)
    )


def fit_clinical_xgb_model(X_train, y_train, X_test):
    search_df = tune_clinical_xgb_model(X_train, y_train)
    best_row = conservative_top_row(
        search_df,
        score_cols=["cv_auc", "cv_auprc"],
        tolerance=OOF_TOLERANCE,
        prefer_cols=["max_depth", "n_estimators"],
        ascending=[True, True],
    )
    best_params = best_row["params"]

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)

    preds = []
    importances = []
    for seed in CLINICAL_XGB_FINAL_SEEDS:
        model = make_xgb_model(best_params, seed, scale_pos_weight)
        model.fit(X_train, y_train)
        preds.append(model.predict_proba(X_test)[:, 1])
        importances.append(model.feature_importances_)

    test_probs = np.mean(np.vstack(preds), axis=0)

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Mean_Importance": np.mean(np.vstack(importances), axis=0),
        "SD_Importance": np.std(np.vstack(importances), axis=0),
    }).sort_values("Mean_Importance", ascending=False).reset_index(drop=True)

    return {
        "search_df": search_df.drop(columns=["params"], errors="ignore"),
        "best_params": best_params,
        "importance_df": importance_df,
        "test_probs": test_probs,
        "best_params_json": best_row["params_json"],
    }


##### 8. BLENDING ##### 

def blend_predictions(clinical_probs, genetic_probs, clinical_weight):
    blended = (
        clinical_weight * np.asarray(clinical_probs)
        + (1.0 - clinical_weight) * np.asarray(genetic_probs)
    )
    return np.clip(blended, 0.0, 1.0)


def tune_static_blend(clinical_oof, genetic_oof, y_train, weight_grid):
    rows = []
    for weight in weight_grid:
        blended = blend_predictions(clinical_oof, genetic_oof, weight)
        rows.append({
            "clinical_weight": weight,
            "genetic_weight": 1.0 - weight,
            "train_oof_auc": roc_auc_score(y_train, blended),
            "train_oof_auprc": average_precision_score(y_train, blended),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["train_oof_auc", "train_oof_auprc", "clinical_weight"],
                     ascending=[False, False, False])
        .reset_index(drop=True)
    )


def _compute_blended_contrib_df(
    feature_df,
    feature_col,
    importance_col,
    final_model_name,
    best_static_weight,
):
    feat = feature_df.copy()
    total = feat[importance_col].sum()

    if total > 0:
        feat["Normalized_Importance"] = feat[importance_col] / total
    else:
        feat["Normalized_Importance"] = 0.0

    if final_model_name == "Static blend":
        clin_w = best_static_weight
        gen_w = 1.0 - best_static_weight
    else:
        clin_w = 1.0
        gen_w = 0.0

    rows = [{
        "Feature": "Genetic_PRS",
        "Relative_Contribution": gen_w,
    }]

    for _, row in feat.iterrows():
        rows.append({
            "Feature": row[feature_col],
            "Relative_Contribution": row["Normalized_Importance"] * clin_w,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Relative_Contribution", ascending=False)
        .reset_index(drop=True)
    )


##### 9. OOF-BYGGARE MED OUTER CV #####
# Clinical preprocessing fit:as på fold-train och transformeras på fold-validation
# Kom ihåg diskussion - undviker all typ av data leakage från validationdelen, även i pre-processningen. Därför fit:as KNN-imputern och scalern endast på trainingdelen i varje fold, och transformeras sedan på test/short follow-up.


def build_oof_clinical_lr_score(train_df, y_train, clinical_cols):
    cv = RepeatedStratifiedKFold(
        n_splits=OUTER_CV_SPLITS,
        n_repeats=OUTER_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    oof_matrix = np.full((len(y_train), OUTER_CV_SPLITS * OUTER_CV_REPEATS), np.nan)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(train_df, y_train)):
        fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[va_idx].reset_index(drop=True)
        y_tr = y_train[tr_idx]

        X_tr, X_va, _ = preprocess_clinical_train_test(
            fold_train_df,
            fold_valid_df,
            clinical_cols,
            n_neighbors=KNN_NEIGHBORS,
        )

        best_row = conservative_top_row(
            tune_clinical_model(X_tr, y_tr),
            score_cols=["cv_auc", "cv_auprc"],
            tolerance=OOF_TOLERANCE,
            prefer_cols=["C", "l1_ratio"],
            ascending=[True, True],
        )

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=best_row["C"],
            l1_ratio=best_row["l1_ratio"],
            max_iter=10000,
            random_state=RANDOM_STATE + fold_id,
        )
        model.fit(X_tr, y_tr)
        oof_matrix[va_idx, fold_id] = model.predict_proba(X_va)[:, 1]

    return np.nanmean(oof_matrix, axis=1)


def build_oof_clinical_rf_score(train_df, y_train, clinical_cols):
    cv = RepeatedStratifiedKFold(
        n_splits=OUTER_CV_SPLITS,
        n_repeats=OUTER_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    oof_matrix = np.full((len(y_train), OUTER_CV_SPLITS * OUTER_CV_REPEATS), np.nan)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(train_df, y_train)):
        fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[va_idx].reset_index(drop=True)
        y_tr = y_train[tr_idx]

        X_tr, X_va, _ = preprocess_clinical_train_test(
            fold_train_df,
            fold_valid_df,
            clinical_cols,
            n_neighbors=KNN_NEIGHBORS,
        )

        best_row = conservative_top_row(
            tune_clinical_rf_model(X_tr, y_tr),
            score_cols=["cv_auc", "cv_auprc"],
            tolerance=OOF_TOLERANCE,
            prefer_cols=["max_depth", "min_samples_leaf"],
            ascending=[True, False],
        )
        best_params = best_row["params"]

        preds = []
        for seed in CLINICAL_RF_FINAL_SEEDS:
            model = make_rf_model(best_params, seed + fold_id)
            model.fit(X_tr, y_tr)
            preds.append(model.predict_proba(X_va)[:, 1])

        oof_matrix[va_idx, fold_id] = np.mean(np.vstack(preds), axis=0)

    return np.nanmean(oof_matrix, axis=1)


def build_oof_clinical_xgb_score(train_df, y_train, clinical_cols):
    cv = RepeatedStratifiedKFold(
        n_splits=OUTER_CV_SPLITS,
        n_repeats=OUTER_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    oof_matrix = np.full((len(y_train), OUTER_CV_SPLITS * OUTER_CV_REPEATS), np.nan)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(train_df, y_train)):
        fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[va_idx].reset_index(drop=True)
        y_tr = y_train[tr_idx]

        X_tr, X_va, _ = preprocess_clinical_train_test(
            fold_train_df,
            fold_valid_df,
            clinical_cols,
            n_neighbors=KNN_NEIGHBORS,
        )

        best_row = conservative_top_row(
            tune_clinical_xgb_model(X_tr, y_tr),
            score_cols=["cv_auc", "cv_auprc"],
            tolerance=OOF_TOLERANCE,
            prefer_cols=["max_depth", "n_estimators"],
            ascending=[True, True],
        )
        best_params = best_row["params"]

        n_pos = np.sum(y_tr == 1)
        n_neg = np.sum(y_tr == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)

        preds = []
        for seed in CLINICAL_XGB_FINAL_SEEDS:
            model = make_xgb_model(best_params, seed + fold_id, scale_pos_weight)
            model.fit(X_tr, y_tr)
            preds.append(model.predict_proba(X_va)[:, 1])

        oof_matrix[va_idx, fold_id] = np.mean(np.vstack(preds), axis=0)

    return np.nanmean(oof_matrix, axis=1)


##### 10. SLUTLIGA FUSIONPIPELINES #####
# Train preprocessas med fit, test med transform från samma train-fitade objekt


def run_lr_combined_pipeline(
    train_df,
    test_df,
    y_train,
    y_test,
    genetic_oof_score,
    genetic_test_score,
    clinical_cols,
):
    Xc_train, Xc_test, clinical_preprocessor = preprocess_clinical_train_test(
        train_df,
        test_df,
        clinical_cols,
        n_neighbors=KNN_NEIGHBORS,
    )

    clinical_result = fit_clinical_model(Xc_train, y_train, Xc_test)
    clinical_oof = build_oof_clinical_lr_score(train_df, y_train, clinical_cols)

    static_search_df = tune_static_blend(
        clinical_oof, genetic_oof_score, y_train, STATIC_WEIGHT_GRID
    )
    best_static = conservative_top_row(
        static_search_df,
        score_cols=["train_oof_auc", "train_oof_auprc"],
        tolerance=OOF_TOLERANCE,
        prefer_cols=["clinical_weight"],
        ascending=[False],
    )
    best_static_weight = float(best_static["clinical_weight"])

    static_oof = blend_predictions(clinical_oof, genetic_oof_score, best_static_weight)
    static_test_probs = blend_predictions(clinical_result["test_probs"], genetic_test_score, best_static_weight)

    candidate_oof_df = pd.DataFrame([
        {"Model": "Clinical only", "OOF_AUROC": roc_auc_score(y_train, clinical_oof), "OOF_AUPRC": average_precision_score(y_train, clinical_oof)},
        {"Model": "Static blend", "OOF_AUROC": roc_auc_score(y_train, static_oof), "OOF_AUPRC": average_precision_score(y_train, static_oof)},
    ]).sort_values(["OOF_AUROC", "OOF_AUPRC"], ascending=[False, False]).reset_index(drop=True)

    final_choice = choose_final_fusion_candidate(candidate_oof_df, tolerance=OOF_TOLERANCE)
    final_model_name = final_choice["Model"]
    selected_oof_row = get_selected_oof_row(candidate_oof_df, final_model_name)

    if final_model_name == "Static blend":
        final_probs = static_test_probs
    else:
        final_probs = clinical_result["test_probs"]

    pred_df = pd.DataFrame({
        "IID": test_df[IID_COL].values,
        "y_test": y_test,
        "Genetic_prob": genetic_test_score,
        "Clinical_prob": clinical_result["test_probs"],
        "Static_blend_prob": static_test_probs,
        "Final_prob": final_probs,
    })

    lr_contribution_df = _compute_blended_contrib_df(
        clinical_result["coef_df"],
        "Feature",
        "Abs_Coefficient",
        final_model_name,
        best_static_weight,
    )

    return {
        "candidate_oof_df": candidate_oof_df,
        "selected_oof_row": selected_oof_row,
        "selected_oof_auc": float(selected_oof_row["OOF_AUROC"]),
        "selected_oof_auprc": float(selected_oof_row["OOF_AUPRC"]),
        "pred_df": pred_df,
        "clinical_search_df": clinical_result["search_df"],
        "clinical_coef_df": clinical_result["coef_df"].copy(),
        "clinical_preprocessor_info": {
            "binary_cols": clinical_preprocessor["binary_cols"],
            "continuous_cols": clinical_preprocessor["continuous_cols"],
            "output_columns": clinical_preprocessor["output_columns"],
        },
        "lr_contribution_df": lr_contribution_df,
        "static_search_df": static_search_df,
        "best_static_weight": best_static_weight,
        "final_model_name": final_model_name,
        "final_probs": final_probs,
        "final_auc": roc_auc_score(y_test, final_probs),
        "final_auprc": average_precision_score(y_test, final_probs),
        "clinical_oof_auc": roc_auc_score(y_train, clinical_oof),
        "clinical_oof_auprc": average_precision_score(y_train, clinical_oof),
        "static_oof_auc": roc_auc_score(y_train, static_oof),
        "static_oof_auprc": average_precision_score(y_train, static_oof),
        "best_clinical_C": clinical_result["best_C"],
        "best_clinical_l1_ratio": clinical_result["best_l1_ratio"],
    }


def run_rf_combined_pipeline(
    train_df,
    test_df,
    y_train,
    y_test,
    genetic_oof_score,
    genetic_test_score,
    clinical_cols,
):
    Xc_train, Xc_test, clinical_preprocessor = preprocess_clinical_train_test(
        train_df,
        test_df,
        clinical_cols,
        n_neighbors=KNN_NEIGHBORS,
    )

    clinical_result = fit_clinical_rf_model(Xc_train, y_train, Xc_test)
    clinical_oof = build_oof_clinical_rf_score(train_df, y_train, clinical_cols)

    static_search_df = tune_static_blend(
        clinical_oof, genetic_oof_score, y_train, STATIC_WEIGHT_GRID
    )
    best_static = conservative_top_row(
        static_search_df,
        score_cols=["train_oof_auc", "train_oof_auprc"],
        tolerance=OOF_TOLERANCE,
        prefer_cols=["clinical_weight"],
        ascending=[False],
    )
    best_static_weight = float(best_static["clinical_weight"])

    static_oof = blend_predictions(clinical_oof, genetic_oof_score, best_static_weight)
    static_test_probs = blend_predictions(clinical_result["test_probs"], genetic_test_score, best_static_weight)

    candidate_oof_df = pd.DataFrame([
        {"Model": "Clinical only", "OOF_AUROC": roc_auc_score(y_train, clinical_oof), "OOF_AUPRC": average_precision_score(y_train, clinical_oof)},
        {"Model": "Static blend", "OOF_AUROC": roc_auc_score(y_train, static_oof), "OOF_AUPRC": average_precision_score(y_train, static_oof)},
    ]).sort_values(["OOF_AUROC", "OOF_AUPRC"], ascending=[False, False]).reset_index(drop=True)

    final_choice = choose_final_fusion_candidate(candidate_oof_df, tolerance=OOF_TOLERANCE)
    final_model_name = final_choice["Model"]
    selected_oof_row = get_selected_oof_row(candidate_oof_df, final_model_name)

    if final_model_name == "Static blend":
        final_probs = static_test_probs
    else:
        final_probs = clinical_result["test_probs"]

    contrib_df = _compute_blended_contrib_df(
        clinical_result["importance_df"],
        "Feature",
        "Mean_Importance",
        final_model_name,
        best_static_weight,
    )

    pred_df = pd.DataFrame({
        "IID": test_df[IID_COL].values,
        "y_test": y_test,
        "Genetic_prob": genetic_test_score,
        "Clinical_prob": clinical_result["test_probs"],
        "Static_blend_prob": static_test_probs,
        "Final_prob": final_probs,
    })

    return {
        "candidate_oof_df": candidate_oof_df,
        "selected_oof_row": selected_oof_row,
        "selected_oof_auc": float(selected_oof_row["OOF_AUROC"]),
        "selected_oof_auprc": float(selected_oof_row["OOF_AUPRC"]),
        "rf_search_df": clinical_result["search_df"],
        "importance_df": clinical_result["importance_df"],
        "clinical_preprocessor_info": {
            "binary_cols": clinical_preprocessor["binary_cols"],
            "continuous_cols": clinical_preprocessor["continuous_cols"],
            "output_columns": clinical_preprocessor["output_columns"],
        },
        "feature_contrib_df": contrib_df,
        "pred_df": pred_df,
        "best_params_json": clinical_result["best_params_json"],
        "best_static_weight": best_static_weight,
        "final_model_name": final_model_name,
        "final_probs": final_probs,
        "final_auc": roc_auc_score(y_test, final_probs),
        "final_auprc": average_precision_score(y_test, final_probs),
    }


def run_xgb_combined_pipeline(
    train_df,
    test_df,
    y_train,
    y_test,
    genetic_oof_score,
    genetic_test_score,
    clinical_cols,
):
    Xc_train, Xc_test, clinical_preprocessor = preprocess_clinical_train_test(
        train_df,
        test_df,
        clinical_cols,
        n_neighbors=KNN_NEIGHBORS,
    )

    clinical_result = fit_clinical_xgb_model(Xc_train, y_train, Xc_test)
    clinical_oof = build_oof_clinical_xgb_score(train_df, y_train, clinical_cols)

    static_search_df = tune_static_blend(
        clinical_oof, genetic_oof_score, y_train, STATIC_WEIGHT_GRID
    )
    best_static = conservative_top_row(
        static_search_df,
        score_cols=["train_oof_auc", "train_oof_auprc"],
        tolerance=OOF_TOLERANCE,
        prefer_cols=["clinical_weight"],
        ascending=[False],
    )
    best_static_weight = float(best_static["clinical_weight"])

    static_oof = blend_predictions(clinical_oof, genetic_oof_score, best_static_weight)
    static_test_probs = blend_predictions(clinical_result["test_probs"], genetic_test_score, best_static_weight)

    candidate_oof_df = pd.DataFrame([
        {"Model": "Clinical only", "OOF_AUROC": roc_auc_score(y_train, clinical_oof), "OOF_AUPRC": average_precision_score(y_train, clinical_oof)},
        {"Model": "Static blend", "OOF_AUROC": roc_auc_score(y_train, static_oof), "OOF_AUPRC": average_precision_score(y_train, static_oof)},
    ]).sort_values(["OOF_AUROC", "OOF_AUPRC"], ascending=[False, False]).reset_index(drop=True)

    final_choice = choose_final_fusion_candidate(candidate_oof_df, tolerance=OOF_TOLERANCE)
    final_model_name = final_choice["Model"]
    selected_oof_row = get_selected_oof_row(candidate_oof_df, final_model_name)

    if final_model_name == "Static blend":
        final_probs = static_test_probs
    else:
        final_probs = clinical_result["test_probs"]

    contrib_df = _compute_blended_contrib_df(
        clinical_result["importance_df"],
        "Feature",
        "Mean_Importance",
        final_model_name,
        best_static_weight,
    )

    pred_df = pd.DataFrame({
        "IID": test_df[IID_COL].values,
        "y_test": y_test,
        "Genetic_prob": genetic_test_score,
        "Clinical_prob": clinical_result["test_probs"],
        "Static_blend_prob": static_test_probs,
        "Final_prob": final_probs,
    })

    return {
        "candidate_oof_df": candidate_oof_df,
        "selected_oof_row": selected_oof_row,
        "selected_oof_auc": float(selected_oof_row["OOF_AUROC"]),
        "selected_oof_auprc": float(selected_oof_row["OOF_AUPRC"]),
        "xgb_search_df": clinical_result["search_df"],
        "importance_df": clinical_result["importance_df"],
        "clinical_preprocessor_info": {
            "binary_cols": clinical_preprocessor["binary_cols"],
            "continuous_cols": clinical_preprocessor["continuous_cols"],
            "output_columns": clinical_preprocessor["output_columns"],
        },
        "feature_contrib_df": contrib_df,
        "pred_df": pred_df,
        "best_params_json": clinical_result["best_params_json"],
        "best_static_weight": best_static_weight,
        "final_model_name": final_model_name,
        "final_probs": final_probs,
        "final_auc": roc_auc_score(y_test, final_probs),
        "final_auprc": average_precision_score(y_test, final_probs),
    }


##### 11. PLOTTNING #####


def _ordered_model_specs(auc_lr, auc_rf, auc_xgb, probs_lr, probs_rf, probs_xgb):
    specs = [
        {"name": "Fusion LR", "auc": auc_lr, "probs": probs_lr, "color": PLOT_BLUE},
        {"name": "Fusion RF", "auc": auc_rf, "probs": probs_rf, "color": PLOT_GREEN},
        {"name": "Fusion XGB", "auc": auc_xgb, "probs": probs_xgb, "color": PLOT_ORANGE},
    ]
    return sorted(specs, key=lambda x: x["auc"])


def show_triple_roc_plot(
    y_true,
    probs_lr,
    auc_lr,
    probs_rf,
    auc_rf,
    probs_xgb,
    auc_xgb,
    figsize=FIGSIZE_LARGE,
    save_path=None,
    show_plot=True,
):
    specs = _ordered_model_specs(auc_lr, auc_rf, auc_xgb, probs_lr, probs_rf, probs_xgb)

    fig, ax = plt.subplots(figsize=figsize)

    for z, spec in enumerate(specs, start=3):
        fpr, tpr, _ = roc_curve(y_true, spec["probs"])
        ax.plot(
            fpr,
            tpr,
            color=spec["color"],
            linewidth=2.6 if spec["name"] == "Fusion RF" else 2.4,
            label=f"{spec['name']} (AUROC = {spec['auc']:.3f})",
            zorder=z,
        )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1.1,
        color="gray",
        alpha=0.9,
        label="Random classifier",
        zorder=2,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate", fontsize=14)
    ax.set_ylabel("True positive rate", fontsize=14)

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("black")

    ax.tick_params(axis="both", width=1.0, length=4, colors="black")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.2, color="gray")

    handles, labels = ax.get_legend_handles_labels()
    best_to_worst = sorted(specs, key=lambda x: x["auc"], reverse=True)
    desired_order = [labels.index(f"{s['name']} (AUROC = {s['auc']:.3f})") for s in best_to_worst]
    desired_order.append(labels.index("Random classifier"))

    legend = ax.legend(
        [handles[i] for i in desired_order],
        [labels[i] for i in desired_order],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=0.95,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.8,
        handletextpad=0.5,
    )
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show_plot:
        plt.show()

    plt.close(fig)


def save_roc_plot_large_only(
    y_true,
    probs_lr,
    auc_lr,
    probs_rf,
    auc_rf,
    probs_xgb,
    auc_xgb,
    out_dir,
    base_name="Fusion_LR_RF_XGB_AUROC",
):
    os.makedirs(out_dir, exist_ok=True)

    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    png_path = os.path.join(out_dir, f"{base_name}.png")

    show_triple_roc_plot(
        y_true,
        probs_lr,
        auc_lr,
        probs_rf,
        auc_rf,
        probs_xgb,
        auc_xgb,
        figsize=FIGSIZE_LARGE,
        save_path=pdf_path,
        show_plot=False,
    )
    show_triple_roc_plot(
        y_true,
        probs_lr,
        auc_lr,
        probs_rf,
        auc_rf,
        probs_xgb,
        auc_xgb,
        figsize=FIGSIZE_LARGE,
        save_path=png_path,
        show_plot=False,
    )

    return pdf_path, png_path


##### 12. LADDA OCH MATCHA DATA #####


print("Laddar data...")

genetic_sheet = pick_sheet_name(GENETIC_SNP_FILE, TARGET_SHEET_CANDIDATES)
clinical_sheet = pick_sheet_name(CLINICAL_FILE, CLINICAL_SHEET_CANDIDATES)

bim, fam, bed = read_plink(PLINK_PREFIX)
fam = fam.copy()
fam["iid_clean"] = fam["iid"].map(clean_id)

df_clin = pd.read_excel(CLINICAL_FILE, sheet_name=clinical_sheet)
df_clin[IID_COL] = df_clin[IID_COL].map(clean_id)

if FOLLOW_UP_COL in df_clin.columns:
    followup_df = df_clin[df_clin[FOLLOW_UP_COL] == 1].copy()
    print(f"Hittade {len(followup_df)} '<6y Follow-up'-patienter.")
else:
    followup_df = pd.DataFrame()
    print(f"VARNING: Kolumnen '{FOLLOW_UP_COL}' saknas.")

df_clin[LN_COL] = to_numeric_safe(df_clin[LN_COL])
df_clin = df_clin[df_clin[LN_COL].isin([0, 1])].copy()

required_cols = [IID_COL, LN_COL, TRAIN_COL, TEST_COL] + CLINICAL_COLS
missing_cols = [col for col in required_cols if col not in df_clin.columns]
if missing_cols:
    raise ValueError(f"Följande kolumner saknas i klinikfilen: {missing_cols}")

iid_to_plink_idx = {iid: i for i, iid in enumerate(fam["iid_clean"])}
df_eligible = df_clin[df_clin[IID_COL].isin(fam["iid_clean"])].copy()
df_eligible["plink_idx"] = df_eligible[IID_COL].map(iid_to_plink_idx)

train_df = df_eligible[df_eligible[TRAIN_COL] == 1].reset_index(drop=True)
test_df = df_eligible[df_eligible[TEST_COL] == 1].reset_index(drop=True)

hq_mask = test_df[CLINICAL_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
hq_test_df = test_df[hq_mask].copy().reset_index(drop=True)
print(f"[INFO] Använder alla {len(hq_test_df)} High Quality patienter.")

hq_iids = set(hq_test_df[IID_COL])

y_train = train_df[LN_COL].astype(int).values
y_test = test_df[LN_COL].astype(int).values

print("--- MATCHNINGSKONTROLL ---")
print(f"Train N: {len(train_df)}")
print(f"Test N:  {len(test_df)}")
print(f"Totalt:  {len(train_df) + len(test_df)}")


##### 13. GENETISK FEATURE-MATRIS ####


print("\nBygger genetiska features...")

selected_snps = load_snp_names(GENETIC_SNP_FILE, genetic_sheet)
bim_snp_names = bim.iloc[:, 1].astype(str).tolist()
usable_snps = [snp for snp in selected_snps if snp in bim_snp_names]

if len(usable_snps) == 0:
    raise ValueError("Inga användbara SNPs hittades i PLINK.")

snp_indices = [bim_snp_names.index(snp) for snp in usable_snps]
genotype_all = bed.compute().astype(float).T

G_train_raw = genotype_all[train_df["plink_idx"].astype(int).values][:, snp_indices]
G_test_raw = genotype_all[test_df["plink_idx"].astype(int).values][:, snp_indices]

geno_imputer = SimpleImputer(strategy="most_frequent")
G_train_imputed = geno_imputer.fit_transform(G_train_raw)
G_test_imputed = geno_imputer.transform(G_test_raw)

Xg_train = build_genetic_features(G_train_imputed, usable_snps)
Xg_test = build_genetic_features(G_test_imputed, usable_snps)

print(f"Antal SNPs: {len(usable_snps)}")
print(f"Antal genetiska features: {Xg_train.shape[1]}")


##### 14. GENETISK RF TILL PRS (GRS) #####

print("\nTränar genetisk RF för PRS...")

genetic_test_score = get_rf_ensemble_probs(
    X_train=Xg_train,
    y_train=y_train,
    X_pred=Xg_test,
    params=BEST_GENETIC_RF_PARAMS,
    seeds=GENETIC_RF_SEEDS,
)

genetic_oof_score = build_oof_rf_score(
    X_train=Xg_train,
    y_train=y_train,
    params=BEST_GENETIC_RF_PARAMS,
    seeds=GENETIC_RF_SEEDS,
)

genetic_prs_test_auc = roc_auc_score(y_test, genetic_test_score)
genetic_prs_test_auprc = average_precision_score(y_test, genetic_test_score)

print(f"Genetic RF PRS Test AUROC: {genetic_prs_test_auc:.4f}")
print(f"Genetic RF PRS Test AUPRC: {genetic_prs_test_auprc:.4f}")


##### 15. FUSION LR #####


print("\nKör Fusion LR pipeline...")
lr_result = run_lr_combined_pipeline(
    train_df=train_df,
    test_df=test_df,
    y_train=y_train,
    y_test=y_test,
    genetic_oof_score=genetic_oof_score,
    genetic_test_score=genetic_test_score,
    clinical_cols=CLINICAL_COLS,
)

print(lr_result["candidate_oof_df"].to_string(index=False))
print(f"\nSelected Fusion LR model from training OOF: {lr_result['final_model_name']}")
print(f"Fusion LR final AUROC: {lr_result['final_auc']:.4f}")
print(f"Fusion LR final AUPRC: {lr_result['final_auprc']:.4f}")


##### 16. FUSION RF #####


print("\nKör Fusion RF pipeline...")
rf_result = run_rf_combined_pipeline(
    train_df=train_df,
    test_df=test_df,
    y_train=y_train,
    y_test=y_test,
    genetic_oof_score=genetic_oof_score,
    genetic_test_score=genetic_test_score,
    clinical_cols=CLINICAL_COLS,
)

print(rf_result["candidate_oof_df"].to_string(index=False))
print(f"\nSelected Fusion RF model from training OOF: {rf_result['final_model_name']}")
print(f"Fusion RF final AUROC: {rf_result['final_auc']:.4f}")
print(f"Fusion RF final AUPRC: {rf_result['final_auprc']:.4f}")
print(f"Best clinical RF params: {rf_result['best_params_json']}")


##### 17. FUSION XGB #####


print("\nKör Fusion XGB pipeline...")
xgb_result = run_xgb_combined_pipeline(
    train_df=train_df,
    test_df=test_df,
    y_train=y_train,
    y_test=y_test,
    genetic_oof_score=genetic_oof_score,
    genetic_test_score=genetic_test_score,
    clinical_cols=CLINICAL_COLS,
)

print(xgb_result["candidate_oof_df"].to_string(index=False))
print(f"\nSelected Fusion XGB model from training OOF: {xgb_result['final_model_name']}")
print(f"Fusion XGB final AUROC: {xgb_result['final_auc']:.4f}")
print(f"Fusion XGB final AUPRC: {xgb_result['final_auprc']:.4f}")
print(f"Best clinical XGB params: {xgb_result['best_params_json']}")


##### 18. HIGH QUALITY TEST-SUBSET #####


print(f"\nUtvärderar modeller på High Quality test-subset (N={len(hq_iids)})...")

if len(hq_iids) == 0:
    raise ValueError("Ingen High Quality-subset kunde byggas.")

lr_pred_hq = lr_result["pred_df"][lr_result["pred_df"]["IID"].isin(hq_iids)].copy().reset_index(drop=True)
rf_pred_hq = rf_result["pred_df"][rf_result["pred_df"]["IID"].isin(hq_iids)].copy().reset_index(drop=True)
xgb_pred_hq = xgb_result["pred_df"][xgb_result["pred_df"]["IID"].isin(hq_iids)].copy().reset_index(drop=True)

lr_hq_auc, lr_hq_auprc = subset_auc_auprc(lr_pred_hq["y_test"].values, lr_pred_hq["Final_prob"].values)
rf_hq_auc, rf_hq_auprc = subset_auc_auprc(rf_pred_hq["y_test"].values, rf_pred_hq["Final_prob"].values)
xgb_hq_auc, xgb_hq_auprc = subset_auc_auprc(xgb_pred_hq["y_test"].values, xgb_pred_hq["Final_prob"].values)

hq_eval_df = pd.DataFrame([
    {"Subset": "High Quality test subset", "Model": f"Fusion LR ({lr_result['final_model_name']})", "Test_AUROC": lr_hq_auc, "Test_AUPRC": lr_hq_auprc},
    {"Subset": "High Quality test subset", "Model": f"Fusion RF ({rf_result['final_model_name']})", "Test_AUROC": rf_hq_auc, "Test_AUPRC": rf_hq_auprc},
    {"Subset": "High Quality test subset", "Model": f"Fusion XGB ({xgb_result['final_model_name']})", "Test_AUROC": xgb_hq_auc, "Test_AUPRC": xgb_hq_auprc},
])

print(hq_eval_df.to_string(index=False))


##### 19. SLUTRESULTAT #####


comparison_df = pd.DataFrame([
    {
        "Subset": "Full test",
        "Model": "Fusion LR",
        "Selected_Model": lr_result["final_model_name"],
        "Selection_basis": "Training OOF with conservative selection",
        "OOF_AUROC": lr_result["selected_oof_auc"],
        "OOF_AUPRC": lr_result["selected_oof_auprc"],
        "Test_AUROC": lr_result["final_auc"],
        "Test_AUPRC": lr_result["final_auprc"],
    },
    {
        "Subset": "Full test",
        "Model": "Fusion RF",
        "Selected_Model": rf_result["final_model_name"],
        "Selection_basis": "Training OOF with conservative selection",
        "OOF_AUROC": rf_result["selected_oof_auc"],
        "OOF_AUPRC": rf_result["selected_oof_auprc"],
        "Test_AUROC": rf_result["final_auc"],
        "Test_AUPRC": rf_result["final_auprc"],
    },
    {
        "Subset": "Full test",
        "Model": "Fusion XGB",
        "Selected_Model": xgb_result["final_model_name"],
        "Selection_basis": "Training OOF with conservative selection",
        "OOF_AUROC": xgb_result["selected_oof_auc"],
        "OOF_AUPRC": xgb_result["selected_oof_auprc"],
        "Test_AUROC": xgb_result["final_auc"],
        "Test_AUPRC": xgb_result["final_auprc"],
    },
]).sort_values(["Test_AUROC", "Test_AUPRC"], ascending=[False, False]).reset_index(drop=True)

print("\n" + "=" * 100)
print("SLUTRESULTAT")
print("=" * 100)
print(comparison_df.to_string(index=False))

winner = comparison_df.iloc[0]["Model"]
print(f"\nBästa modell på fulla test-setet: {winner}")


##### 20. UTVÄRDERING AV anti-dsDNA I FUSION LR #####


print("\n" + "=" * 100)
print("UTVÄRDERING AV anti-dsDNA (FUSION LR)")
print("=" * 100)

clinical_cols_no_dna = [col for col in CLINICAL_COLS if col != "Anti-dsDNA"]

lr_result_no_dna = run_lr_combined_pipeline(
    train_df=train_df,
    test_df=test_df,
    y_train=y_train,
    y_test=y_test,
    genetic_oof_score=genetic_oof_score,
    genetic_test_score=genetic_test_score,
    clinical_cols=clinical_cols_no_dna,
)

print(f"\nFusion LR with anti-dsDNA Test AUROC: {lr_result['final_auc']:.4f}")
print(f"Fusion LR without anti-dsDNA Test AUROC: {lr_result_no_dna['final_auc']:.4f}")

fpr_with, tpr_with, _ = roc_curve(y_test, lr_result["final_probs"])
fpr_without, tpr_without, _ = roc_curve(y_test, lr_result_no_dna["final_probs"])

fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

ax.plot(
    fpr_with, tpr_with,
    color=PLOT_BLUE, linewidth=2.6,
    label=f"Fusion LR with anti-dsDNA (AUROC = {lr_result['final_auc']:.3f})",
    zorder=4,
)
ax.plot(
    fpr_without, tpr_without,
    color=PLOT_GRAY, linewidth=2.4,
    label=f"Fusion LR without anti-dsDNA (AUROC = {lr_result_no_dna['final_auc']:.3f})",
    zorder=3,
)
ax.plot(
    [0, 1], [0, 1],
    linestyle="--", linewidth=1.1, color="gray", alpha=0.9,
    label="Random classifier", zorder=2,
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("False positive rate", fontsize=14)
ax.set_ylabel("True positive rate", fontsize=14)

for side in ["left", "right", "top", "bottom"]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.0)
    ax.spines[side].set_color("black")

ax.tick_params(axis="both", width=1.0, length=4, colors="black")
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.2, color="gray")

handles, labels = ax.get_legend_handles_labels()
desired_order = [
    labels.index(f"Fusion LR with anti-dsDNA (AUROC = {lr_result['final_auc']:.3f})"),
    labels.index(f"Fusion LR without anti-dsDNA (AUROC = {lr_result_no_dna['final_auc']:.3f})"),
    labels.index("Random classifier"),
]
legend = ax.legend(
    [handles[i] for i in desired_order],
    [labels[i] for i in desired_order],
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="lightgray",
    framealpha=0.95,
    borderpad=0.4,
    labelspacing=0.3,
    handlelength=1.8,
    handletextpad=0.5,
)
for text in legend.get_texts():
    text.set_fontfamily("Times New Roman")

plt.tight_layout()

PLOT_DIR = os.path.join(BASE_DIR, "Fusion_model_auroc")
os.makedirs(PLOT_DIR, exist_ok=True)

pdf_path_dna = os.path.join(PLOT_DIR, "Fusion_LR_With_vs_Without_Anti_dsDNA.pdf")
png_path_dna = os.path.join(PLOT_DIR, "Fusion_LR_With_vs_Without_Anti_dsDNA.png")

plt.savefig(pdf_path_dna, bbox_inches="tight")
plt.savefig(png_path_dna, bbox_inches="tight")
plt.show()
plt.close(fig)


##### 21. SPARA UTDATA/RESULTAT #####

OUT_XLSX = os.path.join(BASE_DIR, "FUSION_LR_RF_XGB_train_fitted_KNN.xlsx")

selected_model_summary_df = pd.DataFrame([
    {
        "Fusion_Model": "Fusion LR",
        "Selected_Model": lr_result["final_model_name"],
        "Selected_OOF_AUROC": lr_result["selected_oof_auc"],
        "Selected_OOF_AUPRC": lr_result["selected_oof_auprc"],
        "Final_Test_AUROC": lr_result["final_auc"],
        "Final_Test_AUPRC": lr_result["final_auprc"],
    },
    {
        "Fusion_Model": "Fusion RF",
        "Selected_Model": rf_result["final_model_name"],
        "Selected_OOF_AUROC": rf_result["selected_oof_auc"],
        "Selected_OOF_AUPRC": rf_result["selected_oof_auprc"],
        "Final_Test_AUROC": rf_result["final_auc"],
        "Final_Test_AUPRC": rf_result["final_auprc"],
    },
    {
        "Fusion_Model": "Fusion XGB",
        "Selected_Model": xgb_result["final_model_name"],
        "Selected_OOF_AUROC": xgb_result["selected_oof_auc"],
        "Selected_OOF_AUPRC": xgb_result["selected_oof_auprc"],
        "Final_Test_AUROC": xgb_result["final_auc"],
        "Final_Test_AUPRC": xgb_result["final_auprc"],
    },
])

summary_df = pd.DataFrame([{
    "Genetic_sheet": genetic_sheet,
    "Clinical_sheet": clinical_sheet,
    "KNN_imputation_used": True,
    "KNN_k": KNN_NEIGHBORS,
    "Strict_train_fitted_preprocessing": True,
    "Clinical_imputation_strategy": "KNN fit on train/fold-train and transform applied to test/fold-validation",
    "OOF_conservative_selection": True,
    "Shared_fusion_grids_across_models": True,
    "Shared_inner_cv_budget_across_models": False,
    "Tree_models_use_faster_inner_cv": True,
    "Tree_inner_cv_splits": TREE_INNER_CV_SPLITS,
    "Tree_inner_cv_repeats": TREE_INNER_CV_REPEATS,
    "RF_tuning_seeds_n": len(CLINICAL_RF_TUNING_SEEDS),
    "RF_final_seeds_n": len(CLINICAL_RF_FINAL_SEEDS),
    "XGB_tuning_seeds_n": len(CLINICAL_XGB_TUNING_SEEDS),
    "XGB_final_seeds_n": len(CLINICAL_XGB_FINAL_SEEDS),
    "n_train": len(train_df),
    "n_test": len(test_df),
    "n_hq_test_selected": len(hq_iids),
    "n_snps": len(usable_snps),
    "n_genetic_features": Xg_train.shape[1],

    "Genetic_RF_PRS_Test_AUROC": genetic_prs_test_auc,
    "Genetic_RF_PRS_Test_AUPRC": genetic_prs_test_auprc,

    "LR_Selected_Model": lr_result["final_model_name"],
    "LR_OOF_AUROC": lr_result["selected_oof_auc"],
    "LR_OOF_AUPRC": lr_result["selected_oof_auprc"],
    "LR_Final_Test_AUROC": lr_result["final_auc"],
    "LR_Final_Test_AUPRC": lr_result["final_auprc"],
    "LR_HQ_AUROC": lr_hq_auc,
    "LR_HQ_AUPRC": lr_hq_auprc,
    "LR_Best_Clinical_C": lr_result["best_clinical_C"],
    "LR_Best_Clinical_l1_ratio": lr_result["best_clinical_l1_ratio"],
    "LR_Best_Static_Weight": lr_result["best_static_weight"],
    "LR_Without_Anti_dsDNA_AUROC": lr_result_no_dna["final_auc"],
    "LR_Without_Anti_dsDNA_AUPRC": lr_result_no_dna["final_auprc"],

    "RF_Selected_Model": rf_result["final_model_name"],
    "RF_OOF_AUROC": rf_result["selected_oof_auc"],
    "RF_OOF_AUPRC": rf_result["selected_oof_auprc"],
    "RF_Final_AUROC": rf_result["final_auc"],
    "RF_Final_AUPRC": rf_result["final_auprc"],
    "RF_HQ_AUROC": rf_hq_auc,
    "RF_HQ_AUPRC": rf_hq_auprc,
    "RF_Best_Params_json": rf_result["best_params_json"],

    "XGB_Selected_Model": xgb_result["final_model_name"],
    "XGB_OOF_AUROC": xgb_result["selected_oof_auc"],
    "XGB_OOF_AUPRC": xgb_result["selected_oof_auprc"],
    "XGB_Final_AUROC": xgb_result["final_auc"],
    "XGB_Final_AUPRC": xgb_result["final_auprc"],
    "XGB_HQ_AUROC": xgb_hq_auc,
    "XGB_HQ_AUPRC": xgb_hq_auprc,
    "XGB_Best_Params_json": xgb_result["best_params_json"],
}])

model_parameters_df = pd.DataFrame([
    {
        "Model": "Fusion LR",
        "Selection_method": "Clinical elastic net + conservative late fusion with genetic PRS",
        "Best_model_variant": lr_result["final_model_name"],
        "Best_clinical_C": lr_result["best_clinical_C"],
        "Best_clinical_l1_ratio": lr_result["best_clinical_l1_ratio"],
        "Best_static_weight": lr_result["best_static_weight"],
        "Best_params_json": "",
    },
    {
        "Model": "Fusion RF",
        "Selection_method": "Clinical RF + conservative late fusion with genetic PRS",
        "Best_model_variant": rf_result["final_model_name"],
        "Best_clinical_C": np.nan,
        "Best_clinical_l1_ratio": np.nan,
        "Best_static_weight": rf_result["best_static_weight"],
        "Best_params_json": rf_result["best_params_json"],
    },
    {
        "Model": "Fusion XGB",
        "Selection_method": "Clinical XGB + conservative late fusion with genetic PRS",
        "Best_model_variant": xgb_result["final_model_name"],
        "Best_clinical_C": np.nan,
        "Best_clinical_l1_ratio": np.nan,
        "Best_static_weight": xgb_result["best_static_weight"],
        "Best_params_json": xgb_result["best_params_json"],
    },
])

pred_df = pd.DataFrame({
    "IID": test_df[IID_COL].values,
    "y_test": y_test,
    "Genetic_RF_PRS": genetic_test_score,
    "Fusion_LR_Final_prob": lr_result["final_probs"],
    "Fusion_RF_Final_prob": rf_result["final_probs"],
    "Fusion_XGB_Final_prob": xgb_result["final_probs"],
})

hq_pred_df = pd.DataFrame({
    "IID": lr_pred_hq["IID"].values,
    "y_test": lr_pred_hq["y_test"].values,
    "Fusion_LR_Final_prob": lr_pred_hq["Final_prob"].values,
    "Fusion_RF_Final_prob": rf_pred_hq["Final_prob"].values,
    "Fusion_XGB_Final_prob": xgb_pred_hq["Final_prob"].values,
})

dna_compare_df = pd.DataFrame([
    {"Model": "Fusion LR with anti-dsDNA", "Test_AUROC": lr_result["final_auc"], "Test_AUPRC": lr_result["final_auprc"]},
    {"Model": "Fusion LR without anti-dsDNA", "Test_AUROC": lr_result_no_dna["final_auc"], "Test_AUPRC": lr_result_no_dna["final_auprc"]},
])

clinical_preprocessing_df = pd.DataFrame([
    {
        "Fusion_Model": "Fusion LR",
        "Binary_cols": json.dumps(lr_result["clinical_preprocessor_info"]["binary_cols"]),
        "Continuous_cols": json.dumps(lr_result["clinical_preprocessor_info"]["continuous_cols"]),
        "Output_columns": json.dumps(lr_result["clinical_preprocessor_info"]["output_columns"]),
    },
    {
        "Fusion_Model": "Fusion RF",
        "Binary_cols": json.dumps(rf_result["clinical_preprocessor_info"]["binary_cols"]),
        "Continuous_cols": json.dumps(rf_result["clinical_preprocessor_info"]["continuous_cols"]),
        "Output_columns": json.dumps(rf_result["clinical_preprocessor_info"]["output_columns"]),
    },
    {
        "Fusion_Model": "Fusion XGB",
        "Binary_cols": json.dumps(xgb_result["clinical_preprocessor_info"]["binary_cols"]),
        "Continuous_cols": json.dumps(xgb_result["clinical_preprocessor_info"]["continuous_cols"]),
        "Output_columns": json.dumps(xgb_result["clinical_preprocessor_info"]["output_columns"]),
    },
])

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="1_Summary", index=False)
    comparison_df.to_excel(writer, sheet_name="2_Model_Comparison", index=False)
    hq_eval_df.to_excel(writer, sheet_name="3_HQ_Test_Eval", index=False)
    model_parameters_df.to_excel(writer, sheet_name="4_Model_Parameters", index=False)
    pred_df.to_excel(writer, sheet_name="5_Predictions", index=False)
    hq_pred_df.to_excel(writer, sheet_name="6_HQ_Predictions", index=False)
    dna_compare_df.to_excel(writer, sheet_name="7_Anti_dsDNA_Comparison", index=False)
    selected_model_summary_df.to_excel(writer, sheet_name="8_Selected_Model_Summary", index=False)
    clinical_preprocessing_df.to_excel(writer, sheet_name="9_Clinical_Preprocessing", index=False)

    lr_result["candidate_oof_df"].to_excel(writer, sheet_name="LR_Model_Comparison_OOF", index=False)
    lr_result["clinical_search_df"].to_excel(writer, sheet_name="LR_Clinical_Search", index=False)
    lr_result["clinical_coef_df"].to_excel(writer, sheet_name="LR_Clinical_Coefficients", index=False)
    lr_result["static_search_df"].to_excel(writer, sheet_name="LR_Static_Search", index=False)
    lr_result["lr_contribution_df"].to_excel(writer, sheet_name="LR_Feature_Contrib", index=False)
    lr_result["pred_df"].to_excel(writer, sheet_name="LR_Predictions_Detail", index=False)
    lr_result_no_dna["pred_df"].to_excel(writer, sheet_name="LR_NoDNA_Pred_Detail", index=False)

    rf_result["candidate_oof_df"].to_excel(writer, sheet_name="RF_Model_Comparison_OOF", index=False)
    rf_result["rf_search_df"].to_excel(writer, sheet_name="RF_Clinical_Search", index=False)
    rf_result["importance_df"].to_excel(writer, sheet_name="RF_Clinical_Importance", index=False)
    rf_result["feature_contrib_df"].to_excel(writer, sheet_name="RF_Feature_Contrib", index=False)
    rf_result["pred_df"].to_excel(writer, sheet_name="RF_Predictions_Detail", index=False)

    xgb_result["candidate_oof_df"].to_excel(writer, sheet_name="XGB_Model_Comparison_OOF", index=False)
    xgb_result["xgb_search_df"].to_excel(writer, sheet_name="XGB_Clinical_Search", index=False)
    xgb_result["importance_df"].to_excel(writer, sheet_name="XGB_Clinical_Importance", index=False)
    xgb_result["feature_contrib_df"].to_excel(writer, sheet_name="XGB_Feature_Contrib", index=False)
    xgb_result["pred_df"].to_excel(writer, sheet_name="XGB_Predictions_Detail", index=False)

print(f"\nResultat sparat till:\n{OUT_XLSX}")


##### 22. AUROC-PLOT #####


pdf_path_full, png_path_full = save_roc_plot_large_only(
    y_true=y_test,
    probs_lr=lr_result["final_probs"],
    auc_lr=lr_result["final_auc"],
    probs_rf=rf_result["final_probs"],
    auc_rf=rf_result["final_auc"],
    probs_xgb=xgb_result["final_probs"],
    auc_xgb=xgb_result["final_auc"],
    out_dir=PLOT_DIR,
    base_name="Fusion_LR_RF_XGB_AUROC",
)

print("\nSparad AUROC-plot (Full Test Set):")
print(f"PDF: {pdf_path_full}")
print(f"PNG: {png_path_full}")

show_triple_roc_plot(
    y_true=y_test,
    probs_lr=lr_result["final_probs"],
    auc_lr=lr_result["final_auc"],
    probs_rf=rf_result["final_probs"],
    auc_rf=rf_result["final_auc"],
    probs_xgb=xgb_result["final_probs"],
    auc_xgb=xgb_result["final_auc"],
    figsize=FIGSIZE_LARGE,
    save_path=None,
    show_plot=True,
)

if winner == "Fusion LR":
    hq_probs = lr_pred_hq["Final_prob"].values
    hq_y = lr_pred_hq["y_test"].values
    winner_name = "Fusion LR"
elif winner == "Fusion RF":
    hq_probs = rf_pred_hq["Final_prob"].values
    hq_y = rf_pred_hq["y_test"].values
    winner_name = "Fusion RF"
else:
    hq_probs = xgb_pred_hq["Final_prob"].values
    hq_y = xgb_pred_hq["y_test"].values
    winner_name = "Fusion XGB"

hq_pred_classes = (hq_probs >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(hq_y, hq_pred_classes).ravel()
correct = tp + tn
total = len(hq_y)

print(f"\n*** HIGH QUALITY PRESTANDA ({winner_name}) ***")
print(f"Modellen klassificerade {correct} av {total} personer korrekt ({tp} TP, {tn} TN, {fp} FP och {fn} FN)")
