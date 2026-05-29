# Denna koden jämför de fem intersection-listorna i termer av:
# 1) Hur många SNVs varje lista innehåller
# 2) Hur många SNVs som är gemensamma i alla fem listorna
# OBS - ibland används dsPCs och ibland crazy PCs, men dessa är i det här sammanhanget i princip utbytbara

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from pandas_plink import read_plink
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# # # # # # # # # # # # # # # # # # # #
# 1) KONFIGURATION

BASE_DIR = "/Users/adani406/SLE"
PLINK_PREFIX = os.path.join(BASE_DIR, "G1_S1", "set1mainvars")
EXCEL_FILE = os.path.join(BASE_DIR, "SLE_INTEGRATED_COHORT_EXTRAINFO_FINAL.xlsx")

IID_COL = "IID"
LN_COL = "LN"
TRAIN_COL = "Train"
TEST_COL = "Test"
EXCL_COL = "Excluded"

# # # # # # # # # # # # # # # # # # # #
# FEM INTERSECTION-LISTOR

INTERSECTION_SET_SPECS = {
    "PCs + Gender": {
        "file": os.path.join(
            BASE_DIR,
            "Parallel_Association_RF_Intersection_Union_QC_LD.xlsx"
        ),
        "sheet": "Intersection_Clean"
    },
    "PCs": {
        "file": os.path.join(
            BASE_DIR,
            "Parallel_Association_RF_Intersection_Union_QC_LD_PCsOnly.xlsx"
        ),
        "sheet": "Intersection_Clean"
    },
    "PCs + Gender + Age": {
        "file": os.path.join(
            BASE_DIR,
            "Parallel_Association_RF_Intersection_Union_with_Age_QC_LD.xlsx"
        ),
        "sheet": "Intersection_Clean"
    },
    "PCs + Crazy PCs (with age)": {
        "file": os.path.join(
            BASE_DIR,
            "Parallel_Association_RF_Intersection_Union_StandardPC_plus_CrazyPC_withAgeEmbedded_QC_LD.xlsx"
        ),
        "sheet": "Intersection_Clean"
    },
    "PCs + Crazy PCs (without age)": {
        "file": os.path.join(
            BASE_DIR,
            "Parallel_Association_RF_Intersection_Union_StandardPC_plus_CrazyPC_noAgeEmbedded_QC_LD.xlsx"
        ),
        "sheet": "Intersection_Clean"
    },
}

OUTPUT_XLSX = os.path.join(
    BASE_DIR,
    "Intersection_Only_Comparison_with_PCsOnly_Common5_and_CommonNoAge.xlsx"
)
OUTPUT_UPSET_PNG = os.path.join(
    BASE_DIR,
    "Intersection_Only_UpSet_with_PCsOnly_Common5_and_CommonNoAge.png"
)

RANDOM_STATE = 42
PREFER_XGBOOST = True
MAX_UPSET_INTERSECTIONS = 15

# # # # # # # # # # # # # # # # # # # #
# 2) HJÄLPFUNKTIONER

def clean_id(x):
    return str(x).strip().replace(".0", "")

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def load_selected_snps(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    if "SNP" not in df.columns:
        raise ValueError(f"Filen {xlsx_path}, blad {sheet_name}, saknar kolumnen 'SNP'.")

    snps = (
        df["SNP"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    return snps, df

def build_model():
    """
    XGBoost med default-liknande inställning.
    """
    if PREFER_XGBOOST:
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            return model, "XGBoost_default"
        except Exception:
            pass

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    return model, "RandomForest_fallback"

def evaluate_snv_set(
    set_name,
    snv_list,
    bim_snp_names,
    G_train_all,
    G_test_all,
    train_df,
    test_df
):
    snv_to_idx = {snp: i for i, snp in enumerate(bim_snp_names)}
    usable_snvs = [s for s in snv_list if s in snv_to_idx]

    if len(usable_snvs) == 0:
        return {
            "snv_set": set_name,
            "n_snvs_requested": len(snv_list),
            "n_snvs_used": 0,
            "model_name": "NA",
            "test_auc": np.nan,
            "test_ap": np.nan,
        }

    idx = [snv_to_idx[s] for s in usable_snvs]

    X_tr_gen = G_train_all[:, idx]
    X_te_gen = G_test_all[:, idx]

    imp_gen = SimpleImputer(strategy="most_frequent")
    X_tr_gen_imp = imp_gen.fit_transform(X_tr_gen)
    X_te_gen_imp = imp_gen.transform(X_te_gen)

    y_tr = train_df[LN_COL].astype(int).values
    y_te = test_df[LN_COL].astype(int).values

    model, model_name = build_model()
    model.fit(X_tr_gen_imp, y_tr)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_te_gen_imp)[:, 1]
    else:
        probs = model.predict(X_te_gen_imp)

    test_auc = roc_auc_score(y_te, probs)
    test_ap = average_precision_score(y_te, probs)

    return {
        "snv_set": set_name,
        "n_snvs_requested": len(snv_list),
        "n_snvs_used": len(usable_snvs),
        "model_name": model_name,
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
    }

def build_membership_matrix(snv_sets):
    all_snps = sorted(set().union(*[set(v) for v in snv_sets.values()]))
    membership = pd.DataFrame(index=all_snps)

    for set_name, snps in snv_sets.items():
        membership[set_name] = membership.index.isin(set(snps))

    return membership

def intersection_size_table(membership_df):
    combo_counter = Counter()

    for _, row in membership_df.iterrows():
        active_sets = tuple(membership_df.columns[row.values.astype(bool)])
        combo_counter[active_sets] += 1

    rows = []
    for combo, n in combo_counter.items():
        if len(combo) == 0:
            continue
        rows.append({
            "combination": combo,
            "n_sets_in_combination": len(combo),
            "intersection_size": n,
            "combination_label": " & ".join(combo)
        })

    out = pd.DataFrame(rows).sort_values(
        ["intersection_size", "n_sets_in_combination", "combination_label"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return out

def short_label(name):
    mapping = {
        "PCs": "PCs",
        "PCs + Gender": "PCs + Gender",
        "PCs + Gender + Age": "PCs + Gender + Age",
        "PCs + Crazy PCs (with age)": "PCs + Crazy PCs\n(with age)",
        "PCs + Crazy PCs (without age)": "PCs + Crazy PCs\n(without age)",
    }
    return mapping.get(name, name)

def plot_custom_upset(
    snv_sets,
    output_png,
    title="UpSet diagram of intersection SNV lists",
    max_intersections=15
):
    membership_df = build_membership_matrix(snv_sets)
    set_sizes = membership_df.sum(axis=0).sort_values(ascending=False)
    inter_df = intersection_size_table(membership_df).head(max_intersections).copy()

    set_order = set_sizes.index.tolist()
    set_order_short = [short_label(x) for x in set_order]

    n_sets = len(set_order)
    n_cols = len(inter_df)

    fig = plt.figure(figsize=(max(13, 0.8 * n_cols + 8), 8.0))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[3.4, 5.0],
        height_ratios=[2.7, 2.2],
        wspace=0.20,
        hspace=0.10
    )

    ax_left = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_matrix)

    # # # # # # # # # # # # # # # # # # # #
    # vänster: antal SNPs per scenario

    y_pos = np.arange(n_sets)
    left_values = set_sizes[set_order].values

    ax_left.barh(y_pos, left_values)
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(set_order_short, fontsize=10)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Set size", fontsize=11)
    ax_left.set_title("SNVs per scenario", fontsize=12, pad=8)

    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.tick_params(axis="y", pad=6)

    x_offset = max(left_values) * 0.015 if len(left_values) > 0 else 1
    for i, val in enumerate(left_values):
        ax_left.text(
            val + x_offset,
            i,
            str(int(val)),
            va="center",
            ha="left",
            fontsize=9
        )

    ax_left.set_xlim(0, max(left_values) * 1.18 if len(left_values) > 0 else 1)

    # # # # # # # # # # # # # # # # # # # #
    # topp: intersection sizes

    x_pos = np.arange(n_cols)
    top_values = inter_df["intersection_size"].values

    ax_top.bar(x_pos, top_values)
    ax_top.set_ylabel("Intersection size", fontsize=11)
    ax_top.set_title(title, fontsize=13, pad=10)

    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    y_offset = max(top_values) * 0.015 if len(top_values) > 0 else 1
    for i, val in enumerate(top_values):
        ax_top.text(
            i,
            val + y_offset,
            str(int(val)),
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax_top.set_ylim(0, max(top_values) * 1.12 if len(top_values) > 0 else 1)

    # # # # # # # # # # # # # # # # # # # #
    # matrix
    
    ax_matrix.set_xlim(-0.5, n_cols - 0.5)
    ax_matrix.set_ylim(-0.5, n_sets - 0.5)

    for yi in range(n_sets):
        ax_matrix.scatter(
            x_pos,
            np.repeat(yi, n_cols),
            s=20,
            color="#d0d0d0",
            zorder=1
        )

    for col_idx, combo in enumerate(inter_df["combination"]):
        active_y = [set_order.index(s) for s in combo if s in set_order]

        if len(active_y) > 0:
            ax_matrix.scatter(
                np.repeat(col_idx, len(active_y)),
                active_y,
                s=38,
                color="black",
                zorder=3
            )

        if len(active_y) >= 2:
            ax_matrix.plot(
                [col_idx, col_idx],
                [min(active_y), max(active_y)],
                color="black",
                linewidth=1.2,
                zorder=2
            )

    ax_matrix.set_yticks(np.arange(n_sets))
    ax_matrix.set_yticklabels([""] * n_sets)
    ax_matrix.invert_yaxis()

    ax_matrix.set_xticks(x_pos)
    ax_matrix.set_xticklabels([""] * n_cols)
    ax_matrix.set_xlabel("Observed intersections", fontsize=11)

    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)
    ax_matrix.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return membership_df, inter_df

# # # # # # # # # # # # # # # # # # # #
# 3) LADDA DATA

print("\n==============================")
print("JÄMFÖR INTERSECTION-LISTOR")
print("==============================")

df_clin = pd.read_excel(EXCEL_FILE, sheet_name="Test-Train_Final")
df_clin[IID_COL] = df_clin[IID_COL].map(clean_id)

required_cols = [IID_COL, LN_COL, TRAIN_COL, TEST_COL, EXCL_COL]
missing_cols = [c for c in required_cols if c not in df_clin.columns]
if missing_cols:
    raise ValueError(f"Saknade kolumner i klinisk fil: {missing_cols}")

df_clin[LN_COL] = safe_numeric(df_clin[LN_COL])

df_eligible = df_clin.loc[
    (df_clin[EXCL_COL] != 1) &
    (df_clin[LN_COL].isin([0, 1]))
].copy()

train_df = df_eligible[df_eligible[TRAIN_COL] == 1].copy().reset_index(drop=True)
test_df = df_eligible[df_eligible[TEST_COL] == 1].copy().reset_index(drop=True)

print(f"Train n = {len(train_df)}")
print(f"Test n = {len(test_df)}")

# # # # # # # # # # # # # # # # # # # #
# 4) LADDA PLINK

bim, fam, bed = read_plink(PLINK_PREFIX)

fam = fam.copy()
fam["iid"] = fam["iid"].astype(str).map(clean_id)
bim_snp_names = bim.iloc[:, 1].astype(str).values

G_all = bed.compute().astype(float).T

iid_to_idx = {iid: i for i, iid in enumerate(fam["iid"])}

train_df["plink_idx"] = train_df[IID_COL].map(iid_to_idx)
test_df["plink_idx"] = test_df[IID_COL].map(iid_to_idx)

train_df = train_df.dropna(subset=["plink_idx"]).copy().reset_index(drop=True)
test_df = test_df.dropna(subset=["plink_idx"]).copy().reset_index(drop=True)

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("Train eller test kunde inte matchas mot PLINK.")

train_idx = train_df["plink_idx"].astype(int).values
test_idx = test_df["plink_idx"].astype(int).values

G_train_all = G_all[train_idx, :]
G_test_all = G_all[test_idx, :]

print(f"G_train_all shape: {G_train_all.shape}")
print(f"G_test_all shape: {G_test_all.shape}")

# # # # # # # # # # # # # # # # # # # #
# 5) LADDA DE FEM INTERSECTION-LISTORNA

snv_sets = {}
count_rows = []

for set_name, spec in INTERSECTION_SET_SPECS.items():
    filepath = spec["file"]
    sheet = spec["sheet"]

    if not os.path.exists(filepath):
        print(f"VARNING: Filen saknas och hoppas över: {filepath}")
        continue

    snps, _ = load_selected_snps(filepath, sheet_name=sheet)
    snv_sets[set_name] = snps

    count_rows.append({
        "snv_set": set_name,
        "file": filepath,
        "sheet": sheet,
        "n_snvs": len(snps)
    })

if len(snv_sets) != 5:
    print("\nOBS: Färre än 5 scenarier kunde laddas. Kontrollera filnamnen.\n")

snv_counts_df = pd.DataFrame(count_rows).sort_values("n_snvs", ascending=False).reset_index(drop=True)

print("\nAntal SNVs per scenario:")
print(snv_counts_df.to_string(index=False))

# # # # # # # # # # # # # # # # # # # #
# 6) GEMENSAMMA SNPs I ALLA FEM

all_sets = [set(v) for v in snv_sets.values()]
common_all_five = sorted(list(set.intersection(*all_sets))) if len(all_sets) > 0 else []

print("\nAntal SNPs som förekommer i samtliga laddade scenarier:")
print(len(common_all_five))

common_all_five_df = pd.DataFrame({"SNP": common_all_five})

# # # # # # # # # # # # # # # # # # # #
# 6B) GEMENSAMMA SNPs I ALLA SCENARIER UTAN AGE

no_age_keys = [
    "PCs",
    "PCs + Gender",
    "PCs + Crazy PCs (without age)"
]

available_no_age_sets = [set(snv_sets[k]) for k in no_age_keys if k in snv_sets]

common_no_age = (
    sorted(list(set.intersection(*available_no_age_sets)))
    if len(available_no_age_sets) > 0
    else []
)

print("\nAntal SNVs som överlappar i ALLA SCENARIER UTAN AGE:")
print(len(common_no_age))

common_no_age_df = pd.DataFrame({"SNP": common_no_age})

# # # # # # # # # # # # # # # # # # # #
# 7) PERFORMANCE-JÄMFÖRELSE FÖR DE FEM SCENARIERNA

eval_rows = []

for set_name, snv_list in snv_sets.items():
    res = evaluate_snv_set(
        set_name=set_name,
        snv_list=snv_list,
        bim_snp_names=bim_snp_names,
        G_train_all=G_train_all,
        G_test_all=G_test_all,
        train_df=train_df,
        test_df=test_df
    )
    eval_rows.append(res)

eval_df = pd.DataFrame(eval_rows).sort_values(
    ["test_auc", "test_ap", "n_snvs_used"],
    ascending=[False, False, True]
).reset_index(drop=True)

print("\n==============================")
print("AUROC / AUPRC")
print("==============================")
print(eval_df.to_string(index=False))

# # # # # # # # # # # # # # 
# 7B) PERFORMANCE FÖR ÖVERLAPP ÖVER ALLA FEM SCENARION

common_eval = evaluate_snv_set(
    set_name="COMMON_ALL_FIVE_SCENARIOS",
    snv_list=common_all_five,
    bim_snp_names=bim_snp_names,
    G_train_all=G_train_all,
    G_test_all=G_test_all,
    train_df=train_df,
    test_df=test_df
)

common_eval_df = pd.DataFrame([common_eval])

print("\n==============================")
print("AUROC / AUPRC FÖR GEMENSAMMÄNGDEN I ALLA FEM")
print("==============================")
print(common_eval_df.to_string(index=False))

# # # # # # # # # # # # # # # # # # # #
# 7C) PERFORMANCE FÖR ÖVERLAPP UTAN AGE
common_no_age_eval = evaluate_snv_set(
    set_name="COMMON_NO_AGE_SCENARIOS",
    snv_list=common_no_age,
    bim_snp_names=bim_snp_names,
    G_train_all=G_train_all,
    G_test_all=G_test_all,
    train_df=train_df,
    test_df=test_df
)

common_no_age_eval_df = pd.DataFrame([common_no_age_eval])

print("\n==============================")
print("AUROC / AUPRC FÖR GEMENSAMMÄNGDEN UTAN AGE")
print("==============================")
print(common_no_age_eval_df.to_string(index=False))

# # # # # # # # # # # # # # # # # # # #
# 8) UPSET-DIAGRAM

membership_df, upset_intersections_df = plot_custom_upset(
    snv_sets=snv_sets,
    output_png=OUTPUT_UPSET_PNG,
    title="UpSet diagram of intersection SNV lists",
    max_intersections=MAX_UPSET_INTERSECTIONS
)

print(f"\nUpSet-diagram sparat till:\n{OUTPUT_UPSET_PNG}")

# # # # # # # # # # # # # # # # # # # #
# 9) SPARA TILL EXCEL

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    snv_counts_df.to_excel(writer, sheet_name="SNV_counts", index=False)
    eval_df.to_excel(writer, sheet_name="AUROC_AUPRC", index=False)
    common_all_five_df.to_excel(writer, sheet_name="Common_all_five", index=False)
    common_eval_df.to_excel(writer, sheet_name="Common_all_five_metrics", index=False)
    common_no_age_df.to_excel(writer, sheet_name="Common_no_age", index=False)
    common_no_age_eval_df.to_excel(writer, sheet_name="Common_no_age_metrics", index=False)
    membership_df.reset_index().rename(columns={"index": "SNP"}).to_excel(
        writer, sheet_name="UpSet_membership", index=False
    )
    upset_intersections_df.to_excel(writer, sheet_name="UpSet_intersections", index=False)

print("\n==============================")
print("KLART")
print("==============================")
print(f"Excel sparad till:\n{OUTPUT_XLSX}")
print(f"UpSet-diagram sparat till:\n{OUTPUT_UPSET_PNG}")

