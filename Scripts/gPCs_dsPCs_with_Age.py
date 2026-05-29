import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pandas_plink import read_plink
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")

# # # # # # # # # # #
# 1) KONFIGURATION

BASE_DIR = "/Users/adani406/SLE"
PLINK_PREFIX = os.path.join(BASE_DIR, "G1_S1", "set1mainvars")
EIGENVEC_FILE = os.path.join(BASE_DIR, "G1_S1", "set1mainvars.eigenvec")
EXCEL_FILE = os.path.join(BASE_DIR, "SLE_INTEGRATED_COHORT_EXTRAINFO_FINAL.xlsx")
CANDIDATE_XLSX = os.path.join(BASE_DIR, "G1_S1", "set1vars.xlsx")

# dsPCs med age inbäddat
CRAZY_PCA_FILE = os.path.join(BASE_DIR, "SLUTGILTIG", "Cray_PCA_1_withage.xlsx")

# Output: association
OUTPUT_ASSOC_ALL = os.path.join(
    BASE_DIR,
    "Parallel_Association_All_SNVs_StandardPC_plus_CrazyPC_withAgeEmbedded_QC_LD.xlsx"
)
OUTPUT_ASSOC_SELECTED = os.path.join(
    BASE_DIR,
    "Parallel_Association_Selected_SNVs_StandardPC_plus_CrazyPC_withAgeEmbedded_QC_LD.xlsx"
)

# Output: RF
OUTPUT_RF_ALL = os.path.join(
    BASE_DIR,
    "Parallel_RF_All_SNVs_StandardPC_plus_CrazyPC_withAgeEmbedded_QC_LD.xlsx"
)
OUTPUT_RF_SELECTED = os.path.join(
    BASE_DIR,
    "Parallel_RF_Selected_SNVs_StandardPC_plus_CrazyPC_withAgeEmbedded_QC_LD.xlsx"
)

# Output: QC
OUTPUT_QC_XLSX = os.path.join(
    BASE_DIR,
    "Parallel_QC_Summary_StandardPC_plus_CrazyPC_withAgeEmbedded.xlsx"
)

# Output: intersection / union
OUTPUT_COMBINED_XLSX = os.path.join(
    BASE_DIR,
    "Parallel_Association_RF_Intersection_Union_StandardPC_plus_CrazyPC_withAgeEmbedded_QC_LD.xlsx"
)

IID_COL = "IID"
LN_COL = "LN"
TRAIN_COL = "Train"
EXCL_COL = "Excluded"

# Vanliga/ gPCs
N_PCS = 5
STD_PC_COLS = [f"PC{i}" for i in range(1, N_PCS + 1)]

# dsPCs (med age inbäddat)
CRAZY_PC_COLS = ["PC 1", "PC 2"]

# # # # # # # # # # # # # # # # # # # #
# 2) RESAMPLING

N_SPLITS = 5
N_REPEATS = 20
RANDOM_STATE = 42

# # # # # # # # # # # # # # # # # # # #
# 3) ASSOCIATION: STABILITETSTRÖSKLAR

P_THRESHOLD = 0.05
MIN_SIGNIF_FREQUENCY = 0.40 #HÄR
MIN_DIRECTIONAL_CONSISTENCY = 0.70

REQUIRE_OR_AWAY_FROM_1 = False
MIN_MEDIAN_OR = 1.20
MAX_MEDIAN_OR_FOR_PROTECTIVE = 1 / 1.20

# # # # # # # # # # # # # # # # # # # #
# 4) RF: STABILITETSTRÖSKLAR

RF_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

# Relativ regel för RF när modellen körs på alla QC-passade SNVs
RF_TOP_FRACTION_PER_RUN = 0.10       # top 10% i varje run.  
MIN_RF_IMPORTANCE_FREQUENCY = 0.50

# # # # # # # # # # # # # # # # # # # #
# 5) QC-PARAMETRAR

MIN_MAF = 0.01
MAX_MISSING = 0.05

LD_R2_THRESHOLD = 0.90
LD_WINDOW_SIZE = 200_000

# # # # # # # # # # # # # # # # # # # #
# 6) HJÄLPFUNKTIONER

def clean_id(x):
    return str(x).strip().replace(".0", "")

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def load_candidate_snps(candidate_xlsx):
    cand = pd.read_excel(candidate_xlsx)
    snp_col = cand.columns[0]
    return cand[snp_col].astype(str).tolist()

def load_eigenvec(eigenvec_file, n_pcs=5):
    eig = pd.read_csv(eigenvec_file, sep=r"\s+", header=None)
    eig = eig.copy()
    eig.columns = ["FID", "IID"] + [f"PC{i}" for i in range(1, eig.shape[1] - 1)]
    eig["IID"] = eig["IID"].map(clean_id)

    keep_cols = ["IID"] + [f"PC{i}" for i in range(1, n_pcs + 1)]
    out = eig[keep_cols].copy()

    for c in keep_cols[1:]:
        out[c] = safe_numeric(out[c])

    return out

def load_crazy_pcs(crazy_pca_file, iid_col="IID", pc_cols=None):
    if pc_cols is None:
        pc_cols = ["PC 1", "PC 2"]

    df = pd.read_excel(crazy_pca_file)
    df.columns = [str(c).strip() for c in df.columns]

    if iid_col not in df.columns:
        raise ValueError(f"Crazy PCA-filen saknar kolumnen '{iid_col}'.")

    missing_pc_cols = [c for c in pc_cols if c not in df.columns]
    if len(missing_pc_cols) > 0:
        raise ValueError(f"Crazy PCA-filen saknar PC-kolumner: {missing_pc_cols}")

    out = df[[iid_col] + pc_cols].copy()
    out[iid_col] = out[iid_col].map(clean_id)

    for c in pc_cols:
        out[c] = safe_numeric(out[c])

    return out

def ensure_bim_columns(bim_df):
    bim2 = bim_df.copy()

    if "chrom" not in bim2.columns:
        bim2 = bim2.rename(columns={bim2.columns[0]: "chrom"})
    if "snp" not in bim2.columns:
        bim2 = bim2.rename(columns={bim2.columns[1]: "snp"})
    if "pos" not in bim2.columns:
        if len(bim2.columns) >= 4:
            bim2 = bim2.rename(columns={bim2.columns[3]: "pos"})
        else:
            raise ValueError("Kunde inte identifiera positionskolumn i BIM-filen.")

    return bim2

def compute_maf_and_missing(G):
    miss = np.mean(np.isnan(G), axis=0)
    p = np.nanmean(G, axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)
    maf = np.nan_to_num(maf, nan=0.0)
    return maf, miss

def ld_prune_train(G, bim_df, r2_threshold=0.90, window_size=200_000):
    bim2 = ensure_bim_columns(bim_df).reset_index(drop=True)
    bim2 = bim2.sort_values(["chrom", "pos"]).reset_index()
    old_idx = bim2["index"].values
    G_sorted = G[:, old_idx]

    keep_sorted = np.ones(G_sorted.shape[1], dtype=bool)

    for chrom in bim2["chrom"].unique():
        chrom_mask = bim2["chrom"].values == chrom
        chrom_idx = np.where(chrom_mask)[0]

        for i_local, i in enumerate(chrom_idx):
            if not keep_sorted[i]:
                continue

            xi = G_sorted[:, i]
            xi_ok = ~np.isnan(xi)

            j_stop = min(i_local + 1 + window_size, len(chrom_idx))
            for j_local in range(i_local + 1, j_stop):
                j = chrom_idx[j_local]
                if not keep_sorted[j]:
                    continue

                xj = G_sorted[:, j]
                ok = xi_ok & ~np.isnan(xj)

                if np.sum(ok) < 30:
                    continue

                a = xi[ok]
                b = xj[ok]

                if np.std(a) == 0 or np.std(b) == 0:
                    continue

                r = np.corrcoef(a, b)[0, 1]
                if np.isfinite(r) and (r * r) >= r2_threshold:
                    keep_sorted[j] = False

    keep_original = np.zeros(G.shape[1], dtype=bool)
    keep_original[old_idx[keep_sorted]] = True
    return keep_original

def fit_single_snv_logistic(y, snv, cov_df):
    """
    Logistic regression:
    LN ~ SNV + standard PCs 1-5 + crazy PCs 1-2
    """
    X = cov_df.copy()
    X["SNV"] = snv

    tmp = X.copy()
    tmp["LN"] = y
    tmp = tmp.dropna(axis=0).copy()

    if tmp.shape[0] < 30:
        return np.nan, np.nan, np.nan, np.nan

    if tmp["SNV"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan

    y_fit = tmp["LN"].astype(int).values
    X_fit = tmp.drop(columns=["LN"]).copy()
    X_fit = sm.add_constant(X_fit, has_constant="add")

    try:
        model = sm.Logit(y_fit, X_fit)
        result = model.fit(disp=0)

        beta = result.params["SNV"]
        pval = result.pvalues["SNV"]
        or_val = np.exp(beta)
        direction = np.sign(beta)

        return beta, or_val, pval, direction

    except Exception:
        return np.nan, np.nan, np.nan, np.nan

# # # # # # # # # # # # # # # # # # # #
# 7) LADDA KLINISK DATA

print("\n========================================================")
print("PARALLELLT ASSOCIATION + RF PÅ ALLA QC-PASSADE SNVs")
print("Scenario: standard PCs (1-5) + crazy PCs (1-2, with age embedded)")
print("========================================================")

df_clin = pd.read_excel(EXCEL_FILE, sheet_name="Test-Train_Final")
df_clin.columns = [str(c).strip() for c in df_clin.columns]
df_clin[IID_COL] = df_clin[IID_COL].map(clean_id)

required_cols = [IID_COL, LN_COL, TRAIN_COL, EXCL_COL]
missing_cols = [c for c in required_cols if c not in df_clin.columns]
if len(missing_cols) > 0:
    raise ValueError(f"Saknade kolumner i klinisk fil: {missing_cols}")

df_train = df_clin.loc[
    (df_clin[EXCL_COL] != 1) &
    (df_clin[TRAIN_COL] == 1) &
    (df_clin[LN_COL].notna())
].copy()

df_train[LN_COL] = safe_numeric(df_train[LN_COL])
df_train = df_train[df_train[LN_COL].isin([0, 1])].copy()

print(f"Antal train-individer efter filter: {len(df_train)}")
print("LN-fördelning i train:")
print(df_train[LN_COL].value_counts().sort_index().to_string())

# # # # # # # # # # # # # # # # # # # #
# 8) LADDA gPCs (=standard/vanliga PCs)

df_std_pcs = load_eigenvec(EIGENVEC_FILE, n_pcs=N_PCS)
df_train = df_train.merge(df_std_pcs, on=IID_COL, how="left")

print("Antal saknade vanliga PC-värden efter merge:")
print(df_train[STD_PC_COLS].isna().sum().to_string())

# # # # # # # # # # # # # # # # # # # #
# 9) LADDA dsPCs

df_crazy = load_crazy_pcs(
    crazy_pca_file=CRAZY_PCA_FILE,
    iid_col=IID_COL,
    pc_cols=CRAZY_PC_COLS
)

df_train = df_train.merge(df_crazy, on=IID_COL, how="left")

print("Antal saknade crazy PC-värden efter merge:")
print(df_train[CRAZY_PC_COLS].isna().sum().to_string())

# # # # # # # # # # # # # # # # # # # #
# 10) LADDA PLINK OCH KANDIDAT SNPs

bim, fam, bed = read_plink(PLINK_PREFIX)
bim = ensure_bim_columns(bim)

fam = fam.copy()
fam["iid"] = fam["iid"].astype(str).map(clean_id)

candidate_snps = load_candidate_snps(CANDIDATE_XLSX)
candidate_snps_set = set(candidate_snps)

snp_ids = bim["snp"].astype(str).values
keep_mask = np.isin(snp_ids, list(candidate_snps_set))

if keep_mask.sum() == 0:
    raise ValueError("Inga kandidat-SNVs från Excel hittades i PLINK.")

print(f"Antal kandidat-SNVs i PLINK före QC: {keep_mask.sum()}")

G_all = bed.compute().astype(float).T
G_candidates = G_all[:, keep_mask]
bim_candidates = bim.loc[keep_mask].reset_index(drop=True)

# # # # # # # # # # # # # # # # # # # #
# 11) MATCHA PATIENTER

iid_to_idx = {iid: i for i, iid in enumerate(fam["iid"])}
df_train["plink_idx"] = df_train[IID_COL].map(iid_to_idx)
df_train = df_train.dropna(subset=["plink_idx"]).copy()

if len(df_train) == 0:
    raise ValueError("Inga train-individer kunde matchas mot PLINK.")

plink_idx = df_train["plink_idx"].astype(int).values
G_train_full = G_candidates[plink_idx, :]
y = df_train[LN_COL].astype(int).values

cov_cols = STD_PC_COLS + CRAZY_PC_COLS
cov_df = df_train[cov_cols].copy()

print(f"Trainmatris före QC: {G_train_full.shape[0]} individer x {G_train_full.shape[1]} SNVs")
print(f"Kovariater i urvalet: {cov_cols}")

# # # # # # # # # # # # # # # # # # # #
# 12) FILTERS: QC + MAF + MISSING + LD

maf, miss = compute_maf_and_missing(G_train_full)
qc_mask = (maf >= MIN_MAF) & (miss <= MAX_MISSING)

G_qc = G_train_full[:, qc_mask]
bim_qc = bim_candidates.loc[qc_mask].reset_index(drop=True)

print("\n--- QC-steg ---")
print(f"SNVs före QC: {G_train_full.shape[1]}")
print(f"SNVs efter MAF/missing-filter: {G_qc.shape[1]}")

ld_mask = ld_prune_train(
    G=G_qc,
    bim_df=bim_qc,
    r2_threshold=LD_R2_THRESHOLD,
    window_size=LD_WINDOW_SIZE
)

G_train = G_qc[:, ld_mask]
bim_final = bim_qc.loc[ld_mask].reset_index(drop=True)
snp_names = bim_final["snp"].astype(str).values

print(f"SNVs efter LD-pruning: {G_train.shape[1]}")

qc_summary_df = pd.DataFrame({
    "step": [
        "Candidate SNVs in PLINK",
        "After MAF/missing QC",
        "After LD pruning"
    ],
    "n_snvs": [
        G_train_full.shape[1],
        G_qc.shape[1],
        G_train.shape[1]
    ]
})

qc_detail_df = pd.DataFrame({
    "SNP": bim_candidates["snp"].astype(str).values,
    "maf": maf,
    "missing_rate": miss,
    "pass_maf_missing_qc": qc_mask
})

print(f"\nSlutlig trainmatris efter QC + LD: {G_train.shape[0]} individer x {G_train.shape[1]} SNVs")

# # # # # # # # # # # # # # # # # # # #
# 13) RESAMPLING-UPPSÄTTNING

rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)
n_total_runs = N_SPLITS * N_REPEATS

# # # # # # # # # # # # # # # # # # # #
# 14) ASSOCIATIONSSTEG PÅ ALLA QC-PASSADE SNVs

print("\n========================================================")
print("KÖR ASSOCIATIONSSTEG")
print("========================================================")
print(f"Kör {n_total_runs} resamplings...")

assoc_records = []

for run_id, (sub_idx, _) in enumerate(rskf.split(np.zeros(len(y)), y), start=1):
    print(f"Association resampling {run_id}/{n_total_runs}")

    y_sub = y[sub_idx]
    cov_sub = cov_df.iloc[sub_idx].reset_index(drop=True)
    G_sub = G_train[sub_idx, :]

    for j, snp in enumerate(snp_names):
        beta, or_val, pval, direction = fit_single_snv_logistic(
            y=y_sub,
            snv=G_sub[:, j],
            cov_df=cov_sub
        )

        assoc_records.append({
            "run": run_id,
            "SNP": snp,
            "beta": beta,
            "OR": or_val,
            "p_value": pval,
            "direction": direction,
            "significant": int(pd.notna(pval) and pval < P_THRESHOLD)
        })

assoc_results_df = pd.DataFrame(assoc_records)

assoc_summary_rows = []

for snp, grp in assoc_results_df.groupby("SNP"):
    grp_valid = grp.dropna(subset=["beta", "OR", "p_value"]).copy()

    if len(grp_valid) == 0:
        assoc_summary_rows.append({
            "SNP": snp,
            "n_valid_runs": 0,
            "signif_frequency": np.nan,
            "median_p": np.nan,
            "median_beta": np.nan,
            "median_OR": np.nan,
            "positive_direction_frequency": np.nan,
            "negative_direction_frequency": np.nan,
            "directional_consistency": np.nan,
            "assoc_selected": False
        })
        continue

    pos_freq = np.mean(grp_valid["direction"] > 0)
    neg_freq = np.mean(grp_valid["direction"] < 0)
    dir_consistency = max(pos_freq, neg_freq)

    signif_frequency = np.mean(grp_valid["significant"])
    median_or = np.nanmedian(grp_valid["OR"])
    median_beta = np.nanmedian(grp_valid["beta"])
    median_p = np.nanmedian(grp_valid["p_value"])

    selected = (
        (signif_frequency >= MIN_SIGNIF_FREQUENCY) and
        (dir_consistency >= MIN_DIRECTIONAL_CONSISTENCY)
    )

    if REQUIRE_OR_AWAY_FROM_1:
        selected = selected and (
            (median_or >= MIN_MEDIAN_OR) or
            (median_or <= MAX_MEDIAN_OR_FOR_PROTECTIVE)
        )

    assoc_summary_rows.append({
        "SNP": snp,
        "n_valid_runs": len(grp_valid),
        "signif_frequency": signif_frequency,
        "median_p": median_p,
        "median_beta": median_beta,
        "median_OR": median_or,
        "positive_direction_frequency": pos_freq,
        "negative_direction_frequency": neg_freq,
        "directional_consistency": dir_consistency,
        "assoc_selected": selected
    })

assoc_summary_df = pd.DataFrame(assoc_summary_rows)

geno_stats = pd.DataFrame({
    "SNP": snp_names,
    "genotype_missing_rate": np.mean(np.isnan(G_train), axis=0),
    "mean_genotype": np.nanmean(G_train, axis=0)
})

assoc_summary_df = assoc_summary_df.merge(geno_stats, on="SNP", how="left")

assoc_summary_df = assoc_summary_df.sort_values(
    ["assoc_selected", "signif_frequency", "directional_consistency", "median_p"],
    ascending=[False, False, False, True]
).reset_index(drop=True)

assoc_selected_df = assoc_summary_df[assoc_summary_df["assoc_selected"]].copy()

with pd.ExcelWriter(OUTPUT_ASSOC_ALL, engine="openpyxl") as writer:
    assoc_summary_df.to_excel(writer, sheet_name="Association_summary", index=False)
    assoc_results_df.to_excel(writer, sheet_name="All_runs_long", index=False)

with pd.ExcelWriter(OUTPUT_ASSOC_SELECTED, engine="openpyxl") as writer:
    assoc_selected_df.to_excel(writer, sheet_name="Selected_SNVs", index=False)

print("\nKLART: ASSOCIATIONSSTEG")
print(f"Antal rankade SNVs: {len(assoc_summary_df)}")
print(f"Antal valda SNVs: {len(assoc_selected_df)}")
print(f"Alla associationsresultat sparade till: {OUTPUT_ASSOC_ALL}")
print(f"Valda associations-SNVs sparade till: {OUTPUT_ASSOC_SELECTED}")

# # # # # # # # # # # # # # # # # # # #
# 15) RF-STEG PÅ ALLA QC-PASSADE SNVs

print("\n=====================================================")
print("KÖR RF-STEG")
print("=====================================================")
print(f"Kör {n_total_runs} RF-resamplings...")

X_snp = pd.DataFrame(G_train, columns=snp_names, index=df_train.index)
X_full = pd.concat([cov_df.reset_index(drop=True), X_snp.reset_index(drop=True)], axis=1)

rf_records = []

for run_id, (sub_idx, _) in enumerate(rskf.split(np.zeros(len(y)), y), start=1):
    print(f"RF resampling {run_id}/{n_total_runs}")

    X_sub = X_full.iloc[sub_idx].copy()
    y_sub = y[sub_idx]

    imputer = SimpleImputer(strategy="median")
    X_sub_imp = imputer.fit_transform(X_sub)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_sub_imp, y_sub)

    importances = pd.Series(model.feature_importances_, index=X_full.columns)
    snp_importances = importances.loc[snp_names].copy()

    n_top = max(1, int(np.ceil(len(snp_names) * RF_TOP_FRACTION_PER_RUN)))
    top_snps_this_run = set(snp_importances.sort_values(ascending=False).head(n_top).index)

    for snp in snp_names:
        imp = snp_importances[snp]
        rf_records.append({
            "run": run_id,
            "SNP": snp,
            "importance": imp,
            "important_this_run": int(snp in top_snps_this_run),
            "nonzero": int(imp > 0)
        })

rf_results_df = pd.DataFrame(rf_records)

rf_summary_df = (
    rf_results_df
    .groupby("SNP")
    .agg(
        n_runs=("importance", "count"),
        mean_importance=("importance", "mean"),
        median_importance=("importance", "median"),
        sd_importance=("importance", "std"),
        importance_frequency=("important_this_run", "mean"),
        nonzero_frequency=("nonzero", "mean")
    )
    .reset_index()
)

rf_summary_df = rf_summary_df.merge(geno_stats, on="SNP", how="left")
rf_summary_df["rf_selected"] = rf_summary_df["importance_frequency"] >= MIN_RF_IMPORTANCE_FREQUENCY

rf_summary_df = rf_summary_df.sort_values(
    ["rf_selected", "importance_frequency", "median_importance", "mean_importance"],
    ascending=[False, False, False, False]
).reset_index(drop=True)

rf_selected_df = rf_summary_df[rf_summary_df["rf_selected"]].copy()

with pd.ExcelWriter(OUTPUT_RF_ALL, engine="openpyxl") as writer:
    rf_summary_df.to_excel(writer, sheet_name="RF_summary", index=False)
    rf_results_df.to_excel(writer, sheet_name="All_runs_long", index=False)

with pd.ExcelWriter(OUTPUT_RF_SELECTED, engine="openpyxl") as writer:
    rf_selected_df.to_excel(writer, sheet_name="Selected_SNVs", index=False)

print("\nKLART: RF-STEG")
print(f"Antal rankade SNVs: {len(rf_summary_df)}")
print(f"Antal valda RF-SNVs: {len(rf_selected_df)}")
print(f"Alla RF-resultat sparade till: {OUTPUT_RF_ALL}")
print(f"Valda RF-SNVs sparade till: {OUTPUT_RF_SELECTED}")

# # # # # # # # # # # # # # # # # # # #
# 16) INTERSECTION / UNION 

assoc_selected_set = set(assoc_selected_df["SNP"].astype(str))
rf_selected_set = set(rf_selected_df["SNP"].astype(str))

intersection_snps = sorted(list(assoc_selected_set.intersection(rf_selected_set)))
union_snps = sorted(list(assoc_selected_set.union(rf_selected_set)))
assoc_only_snps = sorted(list(assoc_selected_set - rf_selected_set))
rf_only_snps = sorted(list(rf_selected_set - assoc_selected_set))

intersection_df = pd.DataFrame({"SNP": intersection_snps})
union_df = pd.DataFrame({"SNP": union_snps})
assoc_only_df = pd.DataFrame({"SNP": assoc_only_snps})
rf_only_df = pd.DataFrame({"SNP": rf_only_snps})

summary_df = pd.DataFrame({
    "Metric": [
        "Total SNVs after QC + LD",
        "Selected in association",
        "Selected in RF",
        "Intersection",
        "Union",
        "Association only",
        "RF only"
    ],
    "Value": [
        len(snp_names),
        len(assoc_selected_set),
        len(rf_selected_set),
        len(intersection_snps),
        len(union_snps),
        len(assoc_only_snps),
        len(rf_only_snps)
    ]
})

with pd.ExcelWriter(OUTPUT_COMBINED_XLSX, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    intersection_df.to_excel(writer, sheet_name="Intersection_Clean", index=False)
    union_df.to_excel(writer, sheet_name="Union_Clean", index=False)
    assoc_only_df.to_excel(writer, sheet_name="AssociationOnly_Clean", index=False)
    rf_only_df.to_excel(writer, sheet_name="RFOnly_Clean", index=False)

# # # # # # # # # # # # # # # # # # # #
# 17) SPARA QC

with pd.ExcelWriter(OUTPUT_QC_XLSX, engine="openpyxl") as writer:
    qc_summary_df.to_excel(writer, sheet_name="QC_summary", index=False)
    qc_detail_df.to_excel(writer, sheet_name="QC_detail", index=False)

# # # # # # # # # # # # # # # # # # # #
# 18) SLUTUTSKRIFT

print("\n========================================================")
print("KLART: PARALLELLT ASSOCIATION + RF + INTERSECTION")
print("Scenario: gPCs (1-5) + dsPCs (1-2, with age embedded)")
print("========================================================")
print(summary_df.to_string(index=False))

print("\nFiler sparade:")
print(f"- Association all:      {OUTPUT_ASSOC_ALL}")
print(f"- Association selected: {OUTPUT_ASSOC_SELECTED}")
print(f"- RF all:               {OUTPUT_RF_ALL}")
print(f"- RF selected:          {OUTPUT_RF_SELECTED}")
print(f"- QC summary:           {OUTPUT_QC_XLSX}")
print(f"- Combined output:      {OUTPUT_COMBINED_XLSX}")

