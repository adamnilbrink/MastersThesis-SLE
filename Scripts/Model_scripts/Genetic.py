


import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas_plink import read_plink
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"


##### 1) CONFIGURATION & PATHS ETC #####
BASE_DIR = "/Users/adani406/SLE"
PLINK_PREFIX = os.path.join(BASE_DIR, "G1_S1", "set1mainvars")
CLINICAL_FILE = os.path.join(BASE_DIR, "SLE_INTEGRATED_COHORT_EXTRAINFO_FINAL.xlsx")
SNP_LIST_FILE = os.path.join(BASE_DIR, "Final_SNP_Selection.xlsx")

TARGET_SHEET_CANDIDATES = ["Consensus SNPs", "Consensus_SNVs"]

IID_COL = "IID"
LN_COL = "LN"
TRAIN_COL = "Train"
TEST_COL = "Test"
EXCL_COL = "Excluded"

SUMMARY_FEATURES = [
    "BURDEN_SUM",
    "N_NONZERO",
    "N_HET",
    "N_HOMALT",
    "MEAN_GENO",
    "STD_GENO",
    "MAX_GENO",
    "MIN_GENO",
]

# Same colors used across all plots for consistency and aesthetics 
PLOT_BLUE = "#1167d1"
PLOT_GREEN = "#0e9152"
PLOT_ORANGE = "#f5841a"

RANDOM_STATE = 42

BEST_RF_PARAMS = {
    "n_estimators": 1800,
    "max_depth": 8,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "criterion": "gini",
    "class_weight": "balanced_subsample",
    "max_samples": None,
}

RF_MODEL_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 123]
XGB_MODEL_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 123]

XGB_PARAM_GRID = [
    {
        "n_estimators": 400,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "gamma": 0.2,
    },
    {
        "n_estimators": 600,
        "max_depth": 2,
        "learning_rate": 0.02,
        "min_child_weight": 10,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 1.0,
        "reg_lambda": 4.0,
        "gamma": 0.3,
    },
    {
        "n_estimators": 800,
        "max_depth": 2,
        "learning_rate": 0.015,
        "min_child_weight": 12,
        "subsample": 0.70,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "gamma": 0.4,
    },
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "reg_alpha": 1.0,
        "reg_lambda": 4.0,
        "gamma": 0.4,
    },
    {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.02,
        "min_child_weight": 12,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "reg_alpha": 2.0,
        "reg_lambda": 6.0,
        "gamma": 0.5,
    },
]

XGB_CV_SPLITS = 5
XGB_CV_REPEATS = 5

LR_C_GRID = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
LR_CV_SPLITS = 5
LR_CV_REPEATS = 5

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

FIGSIZE_LARGE = (6.8, 5.1)


##### 2) SIMPLE HELP FUNCTIONS #####
def clean_id(x):
    return str(x).strip().replace(".0", "")


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def find_variant_column(df, sheet_name):
    for col in ["SNP", "SNV"]:
        if col in df.columns:
            return col
    raise ValueError(f"Arket '{sheet_name}' saknar kolumnen 'SNP' eller 'SNV'.")


def resolve_target_sheet(xlsx_path):
    xls = pd.ExcelFile(xlsx_path)
    available = xls.sheet_names

    for name in TARGET_SHEET_CANDIDATES:
        if name in available:
            return name

    raise ValueError(
        "Hittade varken 'Consensus SNPs' eller 'Consensus_SNVs' i Final_SNP_Selection.xlsx."
    )


def load_snp_list(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    variant_col = find_variant_column(df, sheet_name)

    snps = (
        df[variant_col]
        .astype(str)
        .str.strip()
        .replace("nan", np.nan)
        .dropna()
    )
    return snps[snps != ""].drop_duplicates().tolist()


def get_feature_type(feature_name):
    if feature_name.endswith("_DOM"):
        return "Dominant"
    if feature_name.endswith("_REC"):
        return "Recessive"
    if feature_name in SUMMARY_FEATURES:
        return "Summary"
    return "Additive"


def get_parent_snp(feature_name):
    if feature_name.endswith("_DOM"):
        return feature_name[:-4]
    if feature_name.endswith("_REC"):
        return feature_name[:-4]
    return feature_name


def build_baseline_features(X, snp_names):
    X = np.asarray(X, dtype=float)
    base = pd.DataFrame(X, columns=snp_names)

    dom = (base >= 1).astype(float)
    dom.columns = [f"{c}_DOM" for c in base.columns]

    rec = (base == 2).astype(float)
    rec.columns = [f"{c}_REC" for c in base.columns]

    out = pd.concat([base, dom, rec], axis=1)

    out["BURDEN_SUM"] = base.sum(axis=1)
    out["N_NONZERO"] = (base > 0).sum(axis=1).astype(float)
    out["N_HET"] = (base == 1).sum(axis=1).astype(float)
    out["N_HOMALT"] = (base == 2).sum(axis=1).astype(float)
    out["MEAN_GENO"] = base.mean(axis=1)
    out["STD_GENO"] = base.std(axis=1).fillna(0.0)
    out["MAX_GENO"] = base.max(axis=1)
    out["MIN_GENO"] = base.min(axis=1)

    return out


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


def make_lr_model(C):
    return LogisticRegression(
        penalty="l2",
        C=C,
        solver="liblinear",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )


def get_rf_ensemble_probs(X_train, y_train, X_pred, params, seeds):
    all_probs = []

    for seed in seeds:
        model = make_rf_model(params, seed)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_pred)[:, 1]
        all_probs.append(probs)

    return np.mean(np.vstack(all_probs), axis=0)


def get_xgb_ensemble_probs(X_train, y_train, X_pred, params, seeds):
    all_probs = []

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)

    for seed in seeds:
        model = make_xgb_model(params, seed, scale_pos_weight=scale_pos_weight)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_pred)[:, 1]
        all_probs.append(probs)

    return np.mean(np.vstack(all_probs), axis=0)


def get_lr_probs(X_train, y_train, X_pred, C):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    model = make_lr_model(C)
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_pred_scaled)[:, 1]

    return probs, model, scaler


def get_feature_importance_across_rf_ensemble(X_train, y_train, feature_names, params, seeds):
    importance_rows = []

    for seed in seeds:
        model = make_rf_model(params, seed)
        model.fit(X_train, y_train)

        importance_rows.append(
            pd.Series(model.feature_importances_, index=feature_names, name=f"seed_{seed}")
        )

    importance_matrix = pd.DataFrame(importance_rows)

    feature_importance_df = pd.DataFrame({
        "Feature": importance_matrix.columns,
        "Mean_Importance": importance_matrix.mean(axis=0).values,
        "SD_Importance": importance_matrix.std(axis=0).values,
    }).sort_values("Mean_Importance", ascending=False).reset_index(drop=True)

    return feature_importance_df, importance_matrix


def get_feature_importance_across_xgb_ensemble(X_train, y_train, feature_names, params, seeds):
    importance_rows = []

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)

    for seed in seeds:
        model = make_xgb_model(params, seed, scale_pos_weight=scale_pos_weight)
        model.fit(X_train, y_train)

        importance_rows.append(
            pd.Series(model.feature_importances_, index=feature_names, name=f"seed_{seed}")
        )

    importance_matrix = pd.DataFrame(importance_rows)

    feature_importance_df = pd.DataFrame({
        "Feature": importance_matrix.columns,
        "Mean_Importance": importance_matrix.mean(axis=0).values,
        "SD_Importance": importance_matrix.std(axis=0).values,
    }).sort_values("Mean_Importance", ascending=False).reset_index(drop=True)

    return feature_importance_df, importance_matrix


def get_lr_coefficients(X_train, y_train, feature_names, C):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = make_lr_model(C)
    model.fit(X_train_scaled, y_train)

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0],
    })
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df["Direction"] = np.where(
        coef_df["Coefficient"] > 0,
        "Higher value -> higher LN probability",
        np.where(
            coef_df["Coefficient"] < 0,
            "Higher value -> lower LN probability",
            "No directional effect"
        )
    )
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    return coef_df, model, scaler


def build_snp_importance_table(feature_importance_df):
    tmp = feature_importance_df.copy()

    tmp["Feature_Type"] = tmp["Feature"].map(get_feature_type)
    tmp["SNP"] = tmp["Feature"].map(get_parent_snp)
    tmp["Encoding"] = tmp["Feature"].map(get_feature_type)

    snp_only = tmp[tmp["Feature_Type"] != "Summary"].copy()
    summary_only = tmp[tmp["Feature_Type"] == "Summary"].copy()

    additive_df = (
        snp_only[snp_only["Encoding"] == "Additive"][["SNP", "Mean_Importance", "SD_Importance"]]
        .rename(columns={
            "Mean_Importance": "Additive_Importance",
            "SD_Importance": "Additive_SD"
        })
    )

    dominant_df = (
        snp_only[snp_only["Encoding"] == "Dominant"][["SNP", "Mean_Importance", "SD_Importance"]]
        .rename(columns={
            "Mean_Importance": "Dominant_Importance",
            "SD_Importance": "Dominant_SD"
        })
    )

    recessive_df = (
        snp_only[snp_only["Encoding"] == "Recessive"][["SNP", "Mean_Importance", "SD_Importance"]]
        .rename(columns={
            "Mean_Importance": "Recessive_Importance",
            "SD_Importance": "Recessive_SD"
        })
    )

    snp_importance_df = pd.DataFrame({"SNP": sorted(snp_only["SNP"].unique())})
    snp_importance_df = snp_importance_df.merge(additive_df, on="SNP", how="left")
    snp_importance_df = snp_importance_df.merge(dominant_df, on="SNP", how="left")
    snp_importance_df = snp_importance_df.merge(recessive_df, on="SNP", how="left")
    snp_importance_df = snp_importance_df.fillna(0.0)

    snp_importance_df["Total_Importance"] = (
        snp_importance_df["Additive_Importance"]
        + snp_importance_df["Dominant_Importance"]
        + snp_importance_df["Recessive_Importance"]
    )

    snp_importance_df["Best_Encoding"] = snp_importance_df[
        ["Additive_Importance", "Dominant_Importance", "Recessive_Importance"]
    ].idxmax(axis=1).map({
        "Additive_Importance": "Additive",
        "Dominant_Importance": "Dominant",
        "Recessive_Importance": "Recessive"
    })

    snp_importance_df["Rank"] = snp_importance_df["Total_Importance"].rank(
        ascending=False,
        method="min"
    )

    snp_importance_df = snp_importance_df.sort_values(
        ["Total_Importance", "Additive_Importance", "Dominant_Importance", "Recessive_Importance"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    summary_only = summary_only[["Feature", "Mean_Importance", "SD_Importance"]].copy()
    summary_only = summary_only.rename(columns={"Feature": "Summary_Feature"})

    return snp_importance_df, summary_only, tmp


def build_snp_importance_table_from_lr(coef_df):
    tmp = coef_df.copy()
    tmp["Mean_Importance"] = tmp["Abs_Coefficient"]
    tmp["SD_Importance"] = 0.0

    tmp["Feature_Type"] = tmp["Feature"].map(get_feature_type)
    tmp["SNP"] = tmp["Feature"].map(get_parent_snp)
    tmp["Encoding"] = tmp["Feature"].map(get_feature_type)

    snp_only = tmp[tmp["Feature_Type"] != "Summary"].copy()
    summary_only = tmp[tmp["Feature_Type"] == "Summary"].copy()

    additive_df = (
        snp_only[snp_only["Encoding"] == "Additive"][["SNP", "Mean_Importance", "SD_Importance", "Coefficient"]]
        .rename(columns={
            "Mean_Importance": "Additive_Importance",
            "SD_Importance": "Additive_SD",
            "Coefficient": "Additive_Coefficient"
        })
    )

    dominant_df = (
        snp_only[snp_only["Encoding"] == "Dominant"][["SNP", "Mean_Importance", "SD_Importance", "Coefficient"]]
        .rename(columns={
            "Mean_Importance": "Dominant_Importance",
            "SD_Importance": "Dominant_SD",
            "Coefficient": "Dominant_Coefficient"
        })
    )

    recessive_df = (
        snp_only[snp_only["Encoding"] == "Recessive"][["SNP", "Mean_Importance", "SD_Importance", "Coefficient"]]
        .rename(columns={
            "Mean_Importance": "Recessive_Importance",
            "SD_Importance": "Recessive_SD",
            "Coefficient": "Recessive_Coefficient"
        })
    )

    snp_importance_df = pd.DataFrame({"SNP": sorted(snp_only["SNP"].unique())})
    snp_importance_df = snp_importance_df.merge(additive_df, on="SNP", how="left")
    snp_importance_df = snp_importance_df.merge(dominant_df, on="SNP", how="left")
    snp_importance_df = snp_importance_df.merge(recessive_df, on="SNP", how="left")
    snp_importance_df = snp_importance_df.fillna(0.0)

    snp_importance_df["Total_Importance"] = (
        snp_importance_df["Additive_Importance"]
        + snp_importance_df["Dominant_Importance"]
        + snp_importance_df["Recessive_Importance"]
    )

    snp_importance_df["Best_Encoding"] = snp_importance_df[
        ["Additive_Importance", "Dominant_Importance", "Recessive_Importance"]
    ].idxmax(axis=1).map({
        "Additive_Importance": "Additive",
        "Dominant_Importance": "Dominant",
        "Recessive_Importance": "Recessive"
    })

    snp_importance_df["Rank"] = snp_importance_df["Total_Importance"].rank(
        ascending=False,
        method="min"
    )

    snp_importance_df = snp_importance_df.sort_values(
        ["Total_Importance", "Additive_Importance", "Dominant_Importance", "Recessive_Importance"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    summary_only = summary_only[["Feature", "Mean_Importance", "SD_Importance", "Coefficient", "Direction"]].copy()
    summary_only = summary_only.rename(columns={"Feature": "Summary_Feature"})

    return snp_importance_df, summary_only, tmp


def cv_score_xgb(X_train, y_train, params):
    cv = RepeatedStratifiedKFold(
        n_splits=XGB_CV_SPLITS,
        n_repeats=XGB_CV_REPEATS,
        random_state=RANDOM_STATE
    )

    aucs = []
    auprcs = []

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        n_pos = np.sum(y_tr == 1)
        n_neg = np.sum(y_tr == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)

        fold_probs = []
        for seed in XGB_MODEL_SEEDS:
            model = make_xgb_model(
                params=params,
                seed=seed + fold_id,
                scale_pos_weight=scale_pos_weight
            )
            model.fit(X_tr, y_tr)
            fold_probs.append(model.predict_proba(X_va)[:, 1])

        probs = np.mean(np.vstack(fold_probs), axis=0)

        aucs.append(roc_auc_score(y_va, probs))
        auprcs.append(average_precision_score(y_va, probs))

    return {
        "cv_mean_auc": float(np.mean(aucs)),
        "cv_sd_auc": float(np.std(aucs)),
        "cv_mean_auprc": float(np.mean(auprcs)),
        "cv_sd_auprc": float(np.std(auprcs)),
    }


def tune_xgb(X_train, y_train):
    rows = []

    for params in XGB_PARAM_GRID:
        metrics = cv_score_xgb(X_train, y_train, params)
        rows.append({
            "params_json": json.dumps(params, sort_keys=True),
            "params": params,
            **metrics
        })

    search_df = pd.DataFrame(rows).sort_values(
        ["cv_mean_auc", "cv_mean_auprc", "cv_sd_auc"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return search_df


def cv_score_lr(X_train, y_train, C):
    cv = RepeatedStratifiedKFold(
        n_splits=LR_CV_SPLITS,
        n_repeats=LR_CV_REPEATS,
        random_state=RANDOM_STATE
    )

    aucs = []
    auprcs = []

    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        model = make_lr_model(C)
        model.fit(X_tr_scaled, y_tr)
        probs = model.predict_proba(X_va_scaled)[:, 1]

        aucs.append(roc_auc_score(y_va, probs))
        auprcs.append(average_precision_score(y_va, probs))

    return {
        "cv_mean_auc": float(np.mean(aucs)),
        "cv_sd_auc": float(np.std(aucs)),
        "cv_mean_auprc": float(np.mean(auprcs)),
        "cv_sd_auprc": float(np.std(auprcs)),
    }


def tune_lr(X_train, y_train):
    rows = []

    for C in LR_C_GRID:
        metrics = cv_score_lr(X_train, y_train, C)
        rows.append({
            "C": C,
            **metrics
        })

    search_df = pd.DataFrame(rows).sort_values(
        ["cv_mean_auc", "cv_mean_auprc", "cv_sd_auc"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return search_df


def show_triple_beautiful_roc_plot(
    y_true,
    y_prob_lr,
    auc_lr,
    y_prob_rf,
    auc_rf,
    y_prob_xgb,
    auc_xgb,
    figsize=FIGSIZE_LARGE,
    save_path=None,
    show_plot=True
):
    # Model data used for plotting.
    models = [
        {"name": "LR", "prob": y_prob_lr, "auc": auc_lr, "color": PLOT_BLUE, "lw": 2.4},
        {"name": "RF", "prob": y_prob_rf, "auc": auc_rf, "color": PLOT_GREEN, "lw": 2.6},
        {"name": "XGB", "prob": y_prob_xgb, "auc": auc_xgb, "color": PLOT_ORANGE, "lw": 2.4}
    ]

    # Plot the strongest model first in the legend.
    models_sorted = sorted(models, key=lambda x: x["auc"], reverse=True)

    fig, ax = plt.subplots(figsize=figsize)

    for i, mod in enumerate(models_sorted):
        fpr, tpr, _ = roc_curve(y_true, mod["prob"])
        ax.plot(
            fpr,
            tpr,
            color=mod["color"],
            linewidth=mod["lw"],
            label=f"{mod['name']} (AUROC = {mod['auc']:.3f})",
            zorder=5 - i
        )

    # Random classifier baseline.
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.1,
        color="gray",
        alpha=0.9,
        label="Random classifier",
        zorder=2
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate", fontsize=14)
    ax.set_ylabel("True positive rate", fontsize=14)
    ax.set_title("")

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("black")

    ax.tick_params(axis="both", width=1.0, length=4, colors="black")

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.5,
        alpha=0.2,
        color="gray"
    )

    handles, labels = ax.get_legend_handles_labels()
    
    legend = ax.legend(
        handles,
        labels,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=0.95,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.8,
        handletextpad=0.5
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
    y_prob_lr,
    auc_lr,
    y_prob_rf,
    auc_rf,
    y_prob_xgb,
    auc_xgb,
    out_dir,
    base_name="LR_RF_XGB_AUROC_plot_large"
):
    os.makedirs(out_dir, exist_ok=True)

    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    png_path = os.path.join(out_dir, f"{base_name}.png")

    show_triple_beautiful_roc_plot(
        y_true=y_true,
        y_prob_lr=y_prob_lr,
        auc_lr=auc_lr,
        y_prob_rf=y_prob_rf,
        auc_rf=auc_rf,
        y_prob_xgb=y_prob_xgb,
        auc_xgb=auc_xgb,
        figsize=FIGSIZE_LARGE,
        save_path=pdf_path,
        show_plot=False
    )

    show_triple_beautiful_roc_plot(
        y_true=y_true,
        y_prob_lr=y_prob_lr,
        auc_lr=auc_lr,
        y_prob_rf=y_prob_rf,
        auc_rf=auc_rf,
        y_prob_xgb=y_prob_xgb,
        auc_xgb=auc_xgb,
        figsize=FIGSIZE_LARGE,
        save_path=png_path,
        show_plot=False
    )

    return pdf_path, png_path


##### 3) LOAD DATA ####
print("Laddar grunddata...")

bim, fam, bed = read_plink(PLINK_PREFIX)
fam = fam.copy()
fam["iid"] = fam["iid"].map(clean_id)

bim_snp_names = bim.iloc[:, 1].astype(str).tolist()
G_all = bed.compute().astype(float).T

df_clin = pd.read_excel(CLINICAL_FILE, sheet_name="Test-Train_Final")
df_clin[IID_COL] = df_clin[IID_COL].map(clean_id)
df_clin[LN_COL] = safe_numeric(df_clin[LN_COL])

if EXCL_COL in df_clin.columns:
    df_clin = df_clin[(df_clin[EXCL_COL] != 1) | (df_clin[EXCL_COL].isna())].copy()

df_clin = df_clin[df_clin[LN_COL].isin([0, 1])].copy()

iid_to_idx = {iid: i for i, iid in enumerate(fam["iid"])}

df_eligible = df_clin[df_clin[IID_COL].isin(fam["iid"])].copy().reset_index(drop=True)
df_eligible["plink_idx"] = df_eligible[IID_COL].map(iid_to_idx)

if df_eligible.empty:
    raise ValueError("Inga individer kunde matchas mellan klinisk data och PLINK.")

p_idx = df_eligible["plink_idx"].astype(int).values
y_all = df_eligible[LN_COL].astype(int).values

train_mask = (df_eligible[TRAIN_COL] == 1).values
test_mask = (df_eligible[TEST_COL] == 1).values

if train_mask.sum() == 0 or test_mask.sum() == 0:
    raise ValueError("Train eller test blev tomt efter matchning mot PLINK.")

target_sheet = resolve_target_sheet(SNP_LIST_FILE)
print(f"Använder ark: {target_sheet}")

snp_list = load_snp_list(SNP_LIST_FILE, target_sheet)
usable_snps = [s for s in snp_list if s in set(bim_snp_names)]

if len(usable_snps) == 0:
    raise ValueError("Inga användbara SNPs hittades i PLINK för consensus-listan.")

snp_idx = [bim_snp_names.index(s) for s in usable_snps]
snp_names_current = [bim_snp_names[i] for i in snp_idx]

G_filtered = G_all[p_idx, :][:, snp_idx]

X_train_raw = G_filtered[train_mask]
y_train = y_all[train_mask]

X_test_raw = G_filtered[test_mask]
y_test = y_all[test_mask]

print(f"Antal SNPs: {len(snp_names_current)}")
print(f"Train N: {len(y_train)}")
print(f"Test N: {len(y_test)}")


##### 4) IMPUTATION AND FEATURE ENGINEERING #####
print("\nBygger features...")

imp = SimpleImputer(strategy="most_frequent")
X_train_imp = imp.fit_transform(X_train_raw)
X_test_imp = imp.transform(X_test_raw)

X_train = build_baseline_features(X_train_imp, snp_names_current)
X_test = build_baseline_features(X_test_imp, snp_names_current)

print(f"Antal features efter expansion: {X_train.shape[1]}")


##### 5) RANDOM FOREST - RF #####
print("\nTränar RF-ensemble...")

rf_test_probs = get_rf_ensemble_probs(
    X_train=X_train,
    y_train=y_train,
    X_pred=X_test,
    params=BEST_RF_PARAMS,
    seeds=RF_MODEL_SEEDS,
)

rf_test_auc = roc_auc_score(y_test, rf_test_probs)
rf_test_auprc = average_precision_score(y_test, rf_test_probs)

rf_feature_importance_df, rf_importance_matrix = get_feature_importance_across_rf_ensemble(
    X_train=X_train,
    y_train=y_train,
    feature_names=X_train.columns.tolist(),
    params=BEST_RF_PARAMS,
    seeds=RF_MODEL_SEEDS,
)

rf_snp_importance_df, rf_summary_feature_df, rf_parsed_feature_df = build_snp_importance_table(
    rf_feature_importance_df
)


##### 6) LOGISTIC REGRESSION - LR ####
print("\nTunar Logistic Regression train-only...")

lr_search_df = tune_lr(X_train=X_train, y_train=y_train)
best_lr_row = lr_search_df.iloc[0]
best_lr_C = float(best_lr_row["C"])

print("\nBästa Logistic Regression-konfiguration")
print("-" * 80)
print(f"CV mean AUROC: {best_lr_row['cv_mean_auc']:.4f}")
print(f"CV mean AUPRC: {best_lr_row['cv_mean_auprc']:.4f}")
print(f"C: {best_lr_C}")

print("\nTränar Logistic Regression...")

lr_test_probs, lr_model, lr_scaler = get_lr_probs(
    X_train=X_train,
    y_train=y_train,
    X_pred=X_test,
    C=best_lr_C,
)

lr_test_auc = roc_auc_score(y_test, lr_test_probs)
lr_test_auprc = average_precision_score(y_test, lr_test_probs)

lr_coef_df, _, _ = get_lr_coefficients(
    X_train=X_train,
    y_train=y_train,
    feature_names=X_train.columns.tolist(),
    C=best_lr_C,
)

lr_snp_importance_df, lr_summary_feature_df, lr_parsed_feature_df = build_snp_importance_table_from_lr(
    lr_coef_df
)


##### 7) XGBOOST - XGB #####
print("\nTunar XGBoost train-only...")

xgb_search_df = tune_xgb(X_train=X_train, y_train=y_train)
best_xgb_row = xgb_search_df.iloc[0]
best_xgb_params = best_xgb_row["params"]

print("\nBästa XGBoost-konfiguration")
print("-" * 80)
print(f"CV mean AUROC: {best_xgb_row['cv_mean_auc']:.4f}")
print(f"CV mean AUPRC: {best_xgb_row['cv_mean_auprc']:.4f}")
print(f"Params: {best_xgb_row['params_json']}")

print("\nTränar XGBoost-ensemble...")

xgb_test_probs = get_xgb_ensemble_probs(
    X_train=X_train,
    y_train=y_train,
    X_pred=X_test,
    params=best_xgb_params,
    seeds=XGB_MODEL_SEEDS,
)

xgb_test_auc = roc_auc_score(y_test, xgb_test_probs)
xgb_test_auprc = average_precision_score(y_test, xgb_test_probs)

xgb_feature_importance_df, xgb_importance_matrix = get_feature_importance_across_xgb_ensemble(
    X_train=X_train,
    y_train=y_train,
    feature_names=X_train.columns.tolist(),
    params=best_xgb_params,
    seeds=XGB_MODEL_SEEDS,
)

xgb_snp_importance_df, xgb_summary_feature_df, xgb_parsed_feature_df = build_snp_importance_table(
    xgb_feature_importance_df
)


##### 8) RESULTS #####
comparison_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Selection_Basis": "Train CV",
        "Test_AUROC": float(lr_test_auc),
        "Test_AUPRC": float(lr_test_auprc),
        "CV_AUROC": float(best_lr_row["cv_mean_auc"]),
        "CV_AUPRC": float(best_lr_row["cv_mean_auprc"]),
    },
    {
        "Model": "Random Forest ensemble",
        "Selection_Basis": "Fixed parameter ensemble",
        "Test_AUROC": float(rf_test_auc),
        "Test_AUPRC": float(rf_test_auprc),
        "CV_AUROC": np.nan,
        "CV_AUPRC": np.nan,
    },
    {
        "Model": "XGBoost ensemble",
        "Selection_Basis": "Train CV",
        "Test_AUROC": float(xgb_test_auc),
        "Test_AUPRC": float(xgb_test_auprc),
        "CV_AUROC": float(best_xgb_row["cv_mean_auc"]),
        "CV_AUPRC": float(best_xgb_row["cv_mean_auprc"]),
    }
]).sort_values(["Test_AUROC", "Test_AUPRC"], ascending=[False, False]).reset_index(drop=True)

print("\n" + "=" * 90)
print("SLUTRESULTAT")
print("=" * 90)
print(comparison_df.to_string(index=False))

print("\nTopp 15 SNPs enligt LR:")
print(
    lr_snp_importance_df[
        [
            "Rank",
            "SNP",
            "Total_Importance",
            "Best_Encoding",
            "Additive_Importance",
            "Dominant_Importance",
            "Recessive_Importance"
        ]
    ].head(15).to_string(index=False)
)

print("\nTopp 15 SNPs enligt RF:")
print(
    rf_snp_importance_df[
        [
            "Rank",
            "SNP",
            "Total_Importance",
            "Best_Encoding",
            "Additive_Importance",
            "Dominant_Importance",
            "Recessive_Importance"
        ]
    ].head(15).to_string(index=False)
)

print("\nTopp 15 SNPs enligt XGBoost:")
print(
    xgb_snp_importance_df[
        [
            "Rank",
            "SNP",
            "Total_Importance",
            "Best_Encoding",
            "Additive_Importance",
            "Dominant_Importance",
            "Recessive_Importance"
        ]
    ].head(15).to_string(index=False)
)


##### 9) SCIENTIFIC OUTPUT TABLES #####
run_info_df = pd.DataFrame([
    {"Field": "BASE_DIR", "Value": BASE_DIR},
    {"Field": "PLINK_PREFIX", "Value": PLINK_PREFIX},
    {"Field": "CLINICAL_FILE", "Value": CLINICAL_FILE},
    {"Field": "SNP_LIST_FILE", "Value": SNP_LIST_FILE},
    {"Field": "Target_sheet", "Value": target_sheet},
    {"Field": "Random_state", "Value": RANDOM_STATE},
    {"Field": "RF_model_seeds", "Value": json.dumps(RF_MODEL_SEEDS)},
    {"Field": "XGB_model_seeds", "Value": json.dumps(XGB_MODEL_SEEDS)},
    {"Field": "LR_C_grid", "Value": json.dumps(LR_C_GRID)},
    {"Field": "XGB_param_grid", "Value": json.dumps(XGB_PARAM_GRID)},
    {"Field": "XGB_CV_splits", "Value": XGB_CV_SPLITS},
    {"Field": "XGB_CV_repeats", "Value": XGB_CV_REPEATS},
    {"Field": "LR_CV_splits", "Value": LR_CV_SPLITS},
    {"Field": "LR_CV_repeats", "Value": LR_CV_REPEATS},
    {"Field": "RF_fixed_params", "Value": json.dumps(BEST_RF_PARAMS, sort_keys=True)},
])

cohort_summary_df = pd.DataFrame([
    {
        "n_total_eligible": len(df_eligible),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_cases_LN1": int(np.sum(y_train == 1)),
        "train_controls_LN0": int(np.sum(y_train == 0)),
        "test_cases_LN1": int(np.sum(y_test == 1)),
        "test_controls_LN0": int(np.sum(y_test == 0)),
        "case_fraction_train": float(np.mean(y_train)),
        "case_fraction_test": float(np.mean(y_test)),
        "n_selected_snps": len(snp_names_current),
        "n_final_features": X_train.shape[1],
    }
])

selected_snps_df = pd.DataFrame({
    "SNP": snp_names_current,
    "SNP_Order_In_Model_Input": np.arange(1, len(snp_names_current) + 1)
})

feature_inventory_df = pd.DataFrame({
    "Feature": X_train.columns.tolist()
})
feature_inventory_df["Feature_Type"] = feature_inventory_df["Feature"].map(get_feature_type)
feature_inventory_df["Parent_SNP"] = feature_inventory_df["Feature"].map(get_parent_snp)
feature_inventory_df["Is_Summary_Feature"] = feature_inventory_df["Feature_Type"].eq("Summary")
feature_inventory_df["Is_Expanded_Genetic_Encoding"] = ~feature_inventory_df["Is_Summary_Feature"]

summary_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Target_sheet": target_sheet,
        "n_SNPs": len(snp_names_current),
        "n_features": X_train.shape[1],
        "n_models_in_ensemble": 1,
        "Selection_basis": "Train CV",
        "Best_C": best_lr_C,
        "CV_AUROC": float(best_lr_row["cv_mean_auc"]),
        "CV_AUPRC": float(best_lr_row["cv_mean_auprc"]),
        "Test_AUROC": float(lr_test_auc),
        "Test_AUPRC": float(lr_test_auprc),
        "Top_ranked_SNP": lr_snp_importance_df.iloc[0]["SNP"],
        "Top_ranked_encoding": lr_snp_importance_df.iloc[0]["Best_Encoding"],
        "Params_json": json.dumps({"C": best_lr_C, "penalty": "l2", "solver": "liblinear"}, sort_keys=True),
    },
    {
        "Model": "Random Forest ensemble",
        "Target_sheet": target_sheet,
        "n_SNPs": len(snp_names_current),
        "n_features": X_train.shape[1],
        "n_models_in_ensemble": len(RF_MODEL_SEEDS),
        "Selection_basis": "Fixed parameter ensemble",
        "Best_C": np.nan,
        "CV_AUROC": np.nan,
        "CV_AUPRC": np.nan,
        "Test_AUROC": float(rf_test_auc),
        "Test_AUPRC": float(rf_test_auprc),
        "Top_ranked_SNP": rf_snp_importance_df.iloc[0]["SNP"],
        "Top_ranked_encoding": rf_snp_importance_df.iloc[0]["Best_Encoding"],
        "Params_json": json.dumps(BEST_RF_PARAMS, sort_keys=True),
    },
    {
        "Model": "XGBoost ensemble",
        "Target_sheet": target_sheet,
        "n_SNPs": len(snp_names_current),
        "n_features": X_train.shape[1],
        "n_models_in_ensemble": len(XGB_MODEL_SEEDS),
        "Selection_basis": "Train CV",
        "Best_C": np.nan,
        "CV_AUROC": float(best_xgb_row["cv_mean_auc"]),
        "CV_AUPRC": float(best_xgb_row["cv_mean_auprc"]),
        "Test_AUROC": float(xgb_test_auc),
        "Test_AUPRC": float(xgb_test_auprc),
        "Top_ranked_SNP": xgb_snp_importance_df.iloc[0]["SNP"],
        "Top_ranked_encoding": xgb_snp_importance_df.iloc[0]["Best_Encoding"],
        "Params_json": json.dumps(best_xgb_params, sort_keys=True),
    }
])

test_predictions_df = pd.DataFrame({
    "IID": df_eligible.loc[test_mask, IID_COL].values,
    "y_test": y_test,
    "LR_prob": lr_test_probs,
    "RF_prob": rf_test_probs,
    "XGB_prob": xgb_test_probs,
})

test_predictions_df["LR_pred_class_0_5"] = (test_predictions_df["LR_prob"] >= 0.5).astype(int)
test_predictions_df["RF_pred_class_0_5"] = (test_predictions_df["RF_prob"] >= 0.5).astype(int)
test_predictions_df["XGB_pred_class_0_5"] = (test_predictions_df["XGB_prob"] >= 0.5).astype(int)

test_predictions_df["LR_abs_error"] = np.abs(test_predictions_df["y_test"] - test_predictions_df["LR_prob"])
test_predictions_df["RF_abs_error"] = np.abs(test_predictions_df["y_test"] - test_predictions_df["RF_prob"])
test_predictions_df["XGB_abs_error"] = np.abs(test_predictions_df["y_test"] - test_predictions_df["XGB_prob"])

test_predictions_df["Mean_prob_across_models"] = test_predictions_df[["LR_prob", "RF_prob", "XGB_prob"]].mean(axis=1)
test_predictions_df["SD_prob_across_models"] = test_predictions_df[["LR_prob", "RF_prob", "XGB_prob"]].std(axis=1)

lr_top15_df = lr_snp_importance_df.head(15).copy()
rf_top15_df = rf_snp_importance_df.head(15).copy()
xgb_top15_df = xgb_snp_importance_df.head(15).copy()

lr_summary_top_df = lr_summary_feature_df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)
rf_summary_top_df = rf_summary_feature_df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)
xgb_summary_top_df = xgb_summary_feature_df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)

lr_parsed_feature_df = lr_parsed_feature_df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)
rf_parsed_feature_df = rf_parsed_feature_df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)
xgb_parsed_feature_df = xgb_parsed_feature_df.sort_values("Mean_Importance", ascending=False).reset_index(drop=True)


##### 10) SAVE EXCEL OUTPUT #####
OUT_XLSX = os.path.join(BASE_DIR, "Genetic_Models.xlsx")

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    run_info_df.to_excel(writer, sheet_name="00_Run_Info", index=False)
    cohort_summary_df.to_excel(writer, sheet_name="01_Cohort_Summary", index=False)
    summary_df.to_excel(writer, sheet_name="02_Model_Summary", index=False)
    comparison_df.to_excel(writer, sheet_name="03_Model_Comparison", index=False)

    selected_snps_df.to_excel(writer, sheet_name="04_Selected_SNPs", index=False)
    feature_inventory_df.to_excel(writer, sheet_name="05_Feature_Inventory", index=False)
    test_predictions_df.to_excel(writer, sheet_name="06_Test_Predictions", index=False)

    lr_search_df.to_excel(writer, sheet_name="10_LR_CV_Search", index=False)
    lr_coef_df.to_excel(writer, sheet_name="11_LR_Coefficients", index=False)
    lr_parsed_feature_df.to_excel(writer, sheet_name="12_LR_Parsed_Features", index=False)
    lr_snp_importance_df.to_excel(writer, sheet_name="13_LR_SNP_Ranking", index=False)
    lr_summary_feature_df.to_excel(writer, sheet_name="14_LR_Summary_Features", index=False)
    lr_top15_df.to_excel(writer, sheet_name="15_LR_Top15_SNPs", index=False)
    lr_summary_top_df.to_excel(writer, sheet_name="16_LR_Summary_Ranked", index=False)

    rf_feature_importance_df.to_excel(writer, sheet_name="20_RF_All_Features", index=False)
    rf_parsed_feature_df.to_excel(writer, sheet_name="21_RF_Parsed_Features", index=False)
    rf_snp_importance_df.to_excel(writer, sheet_name="22_RF_SNP_Ranking", index=False)
    rf_summary_feature_df.to_excel(writer, sheet_name="23_RF_Summary_Features", index=False)
    rf_importance_matrix.to_excel(writer, sheet_name="24_RF_Seed_Importance", index=True)
    rf_top15_df.to_excel(writer, sheet_name="25_RF_Top15_SNPs", index=False)
    rf_summary_top_df.to_excel(writer, sheet_name="26_RF_Summary_Ranked", index=False)

    xgb_search_df.drop(columns=["params"], errors="ignore").to_excel(writer, sheet_name="30_XGB_CV_Search", index=False)
    xgb_feature_importance_df.to_excel(writer, sheet_name="31_XGB_All_Features", index=False)
    xgb_parsed_feature_df.to_excel(writer, sheet_name="32_XGB_Parsed_Features", index=False)
    xgb_snp_importance_df.to_excel(writer, sheet_name="33_XGB_SNP_Ranking", index=False)
    xgb_summary_feature_df.to_excel(writer, sheet_name="34_XGB_Summary_Features", index=False)
    xgb_importance_matrix.to_excel(writer, sheet_name="35_XGB_Seed_Importance", index=True)
    xgb_top15_df.to_excel(writer, sheet_name="36_XGB_Top15_SNPs", index=False)
    xgb_summary_top_df.to_excel(writer, sheet_name="37_XGB_Summary_Ranked", index=False)

print("\nSparat till:")
print(OUT_XLSX)


##### 11) SAVE AUROC PLOT #####
PLOT_DIR = os.path.join(BASE_DIR, "roc_plot_large_only")

pdf_path, png_path = save_roc_plot_large_only(
    y_true=y_test,
    y_prob_lr=lr_test_probs,
    auc_lr=lr_test_auc,
    y_prob_rf=rf_test_probs,
    auc_rf=rf_test_auc,
    y_prob_xgb=xgb_test_probs,
    auc_xgb=xgb_test_auc,
    out_dir=PLOT_DIR,
    base_name="LR_RF_XGB_AUROC_plot_large"
)

print("\nSparad ROC-plot, LARGE only:")
print(f"PDF: {pdf_path}")
print(f"PNG: {png_path}")

show_triple_beautiful_roc_plot(
    y_true=y_test,
    y_prob_lr=lr_test_probs,
    auc_lr=lr_test_auc,
    y_prob_rf=rf_test_probs,
    auc_rf=rf_test_auc,
    y_prob_xgb=xgb_test_probs,
    auc_xgb=xgb_test_auc,
    figsize=FIGSIZE_LARGE,
    save_path=None,
    show_plot=True
)