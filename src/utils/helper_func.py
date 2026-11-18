import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



 
 

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_binary_metrics(
    y_true,
    y_prob,
    threshold=0.5
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1
    }





def stratified_bootstrap_external_metrics(
    y_true,
    y_prob,
    threshold=0.5,
    n_bootstraps=5000,
    confidence_level=0.95,
    random_state=42
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    rng = np.random.default_rng(random_state)

    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]

    bootstrap_results = []

    for bootstrap_id in range(n_bootstraps):

        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True
        )

        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True
        )

        sampled_indices = np.concatenate([
            sampled_positive,
            sampled_negative
        ])

        rng.shuffle(sampled_indices)

        metrics = calculate_binary_metrics(
            y_true[sampled_indices],
            y_prob[sampled_indices],
            threshold=threshold
        )

        metrics["bootstrap_id"] = bootstrap_id
        bootstrap_results.append(metrics)

    bootstrap_df = pd.DataFrame(bootstrap_results)

    alpha = 1 - confidence_level
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)

    ci_df = pd.DataFrame([
        {
            "metric": metric,
            "ci_lower": np.percentile(
                bootstrap_df[metric].dropna(),
                lower
            ),
            "ci_upper": np.percentile(
                bootstrap_df[metric].dropna(),
                upper
            ),
            "bootstrap_mean":
                bootstrap_df[metric].mean()
        }
        for metric in [
            "AUROC",
            "AUPRC",
            "Sensitivity",
            "Specificity",
            "F1"
        ]
    ])

    return bootstrap_df, ci_df



def bootstrap_external_metrics(
    y_true,
    y_prob,
    threshold=0.5,
    n_bootstraps=5000,
    confidence_level=0.95,
    random_state=42
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must have the same length."
        )

    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)

    bootstrap_results = []
    skipped_samples = 0

    for bootstrap_id in range(n_bootstraps):

        # Draw n_samples indices with replacement
        sampled_indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        y_boot = y_true[sampled_indices]
        prob_boot = y_prob[sampled_indices]

        # AUROC and AUPRC are not meaningful when only one class
        # appears in the resampled cohort.
        if np.unique(y_boot).size < 2:
            skipped_samples += 1
            continue

        metrics = calculate_binary_metrics(
            y_boot,
            prob_boot,
            threshold=threshold
        )

        metrics["bootstrap_id"] = bootstrap_id
        bootstrap_results.append(metrics)

    bootstrap_df = pd.DataFrame(bootstrap_results)

    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    ci_rows = []

    for metric in [
        "AUROC",
        "AUPRC",
        "Sensitivity",
        "Specificity",
        "F1"
    ]:
        values = bootstrap_df[metric].dropna()

        ci_rows.append({
            "metric": metric,
            "bootstrap_mean": values.mean(),
            "bootstrap_median": values.median(),
            "ci_lower": np.percentile(
                values,
                lower_percentile
            ),
            "ci_upper": np.percentile(
                values,
                upper_percentile
            ),
            "valid_bootstraps": len(values)
        })

    ci_df = pd.DataFrame(ci_rows)

    return bootstrap_df, ci_df, skipped_samples




 
# ── Core function ─────────────────────────────────────────────────────────────
 
def feature_significance(
    X: pd.DataFrame,
    y,
    feature_types: dict = None,
    alpha: float = 0.05,
):
    """
    Run univariate significance tests for each feature vs binary class labels.
 
      - Continuous features → Mann-Whitney U test
      - Binary features     → Fisher's exact test
      - All features        → Mutual information (unified ranking, no p-value)
 
    Args:
        X:             DataFrame of features (samples × features).
        y:             Binary class labels (array-like).
        feature_types: Dict mapping feature name → "continuous" or "binary".
                       If None, auto-detected: features with only 0/1 values = binary.
        alpha:         Significance threshold for the significant flag.
 
    Returns:
        pd.DataFrame sorted by p-value (ascending), with columns:
          feature, type, test_used, stat, p_value, mutual_info, significant
    """
    y = np.asarray(y)
 
    # ── Auto-detect feature types ─────────────────────────────────────────────
    if feature_types is None:
        feature_types = {}
        for col in X.columns:
            uniq = X[col].dropna().unique()
            feature_types[col] = "binary" if set(uniq).issubset({0, 1}) else "continuous"
 
    # ── Mutual information (NaNs filled with 0 for sklearn) ──────────────────
    mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)
    mi_map    = dict(zip(X.columns, mi_scores))
 
    # ── Per-feature univariate tests ──────────────────────────────────────────
    rows = []
    for col in X.columns:
        ftype = feature_types.get(col, "continuous")
 
        # Drop NaNs robustly using dropna — works regardless of dtype or None/NaN mix
        col_series = X[col].copy()
        valid_mask = col_series.notna().values          # numpy bool array
        vals       = col_series.dropna().to_numpy(dtype=float if ftype == "continuous" else object)
        y_clean    = y[valid_mask]
        n_nan      = int((~valid_mask).sum())
 
        pos_vals = vals[y_clean == 1]
        neg_vals = vals[y_clean == 0]
 
        try:
            if ftype == "continuous":
                if len(pos_vals) == 0 or len(neg_vals) == 0:
                    raise ValueError("One class has no samples after NaN removal.")
                stat, p = stats.mannwhitneyu(
                    pos_vals.astype(float), neg_vals.astype(float),
                    alternative="two-sided",
                )
                test_used = "Mann-Whitney U"
            else:
                ct = pd.crosstab(vals, y_clean)
                if ct.shape != (2, 2):
                    raise ValueError(f"Crosstab shape {ct.shape} — feature may have only one unique value after NaN removal.")
                _, p = stats.fisher_exact(ct.values)
                stat = float(ct.values[1, 1])   # count: feature=1 & class=1
                test_used = "Fisher's exact"
        except Exception as e:
            print(f"  [WARNING] {col}: {e}")
            stat, p, test_used = np.nan, np.nan, "Failed"
 
        rows.append({
            "feature":     col,
            "type":        ftype,
            "test_used":   test_used,
            "n_nan":       int(n_nan),
            "n_used":      int(valid_mask.sum()),
            "stat":        round(float(stat), 4) if not np.isnan(stat) else np.nan,
            "p_value":     p,
            "mutual_info": round(mi_map[col], 4),
            "significant": p < alpha if not np.isnan(p) else False,
        })
 
    results = pd.DataFrame(rows).sort_values("p_value")
    results["p_value"] = results["p_value"].apply(_fmt_p)
    # Reorder columns for readability
    cols = ["feature", "type", "test_used", "n_used", "n_nan",
            "stat", "p_value", "mutual_info", "significant"]
    return results[cols].reset_index(drop=True)
 
 
def _fmt_p(p):
    if pd.isna(p):
        return np.nan
    if p < 0.001:
        return "<0.001"
    return round(p, 4)


def find_common_features(list1: list[str], list2: list[str], threshold: float = 0.8) -> dict:
    """
    Find common features between two lists of column names, tolerating
    case differences and minor spelling variations.

    Parameters
    ----------
    list1 : list of str
        First list of column names (e.g. from CheckMate).
    list2 : list of str
        Second list of column names (e.g. from E-MTAB).
    threshold : float
        Similarity score (0–1) above which two names are considered a match.
        Default 0.8 is forgiving of case/underscore/minor typos.
        Lower it (e.g. 0.6) for looser matching.

    Returns
    -------
    dict with keys:
        "exact"   — names that matched exactly (case-insensitive)
        "fuzzy"   — names that matched approximately: list of (name1, name2, score)
        "unmatched_list1" — names in list1 with no match in list2
        "unmatched_list2" — names in list2 with no match in list1
    """
    from difflib import SequenceMatcher

    def normalise(s):
        return s.lower().replace("_", "").replace("-", "").replace(" ", "")

    list2_norm = {normalise(col): col for col in list2}

    exact, fuzzy = [], []
    matched_list2 = set()
    unmatched_list1 = []

    for col1 in list1:
        norm1 = normalise(col1)

        # 1. Exact match after normalisation
        if norm1 in list2_norm:
            exact.append((col1, list2_norm[norm1]))
            matched_list2.add(list2_norm[norm1])

        else:
            # 2. Fuzzy match — find best candidate in list2
            best_score, best_col2 = 0, None
            for norm2, col2 in list2_norm.items():
                score = SequenceMatcher(None, norm1, norm2).ratio()
                if score > best_score:
                    best_score, best_col2 = score, col2

            if best_score >= threshold:
                fuzzy.append((col1, best_col2, round(best_score, 3)))
                matched_list2.add(best_col2)
            else:
                unmatched_list1.append(col1)

    unmatched_list2 = [c for c in list2 if c not in matched_list2]

    return {
        "exact": exact,
        "fuzzy": fuzzy,
        "unmatched_list1": unmatched_list1,
        "unmatched_list2": unmatched_list2,
    }



def mahalanobis_outliers(scores, n_pcs=5, alpha=0.001):
    X = scores.iloc[:, :n_pcs].values
    cov = np.cov(X.T)
    inv_cov = np.linalg.inv(cov)
    center = X.mean(axis=0)
    
    distances = [mahalanobis(x, center, inv_cov)**2 for x in X]
    threshold = chi2.ppf(1 - alpha, df=n_pcs)  # e.g. p<0.001
    
    return pd.Series(distances, index=scores.index) > threshold



def compute_pca_scores(
    df: pd.DataFrame,
    n_components: int = 10,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Run PCA on a samples × features DataFrame.

    Parameters
    ----------
    df           : samples as rows, features (genes/signatures) as columns.
                   Index should be sample IDs.
    n_components : number of PCs to compute (default 10).
    scale        : standardise features to zero mean, unit variance before PCA.
                   Almost always True for RNA-seq / mixed-feature data.

    Returns
    -------
    scores       : DataFrame of shape (n_samples, n_components),
                   columns = ["PC1", "PC2", ...], index = sample IDs.
    loadings     : DataFrame of shape (n_features, n_components),
                   how much each feature contributes to each PC.
    pca          : fitted sklearn PCA object (access .explained_variance_ratio_ etc.)
    """


    # Drop columns with zero variance (constant features break scaling)
    df_clean = df.loc[:, df.std() > 0]
    dropped = df.shape[1] - df_clean.shape[1]
    if dropped:
        print(f"Dropped {dropped} zero-variance features before PCA.")

    X = StandardScaler().fit_transform(df_clean) if scale else df_clean.values

    pca = PCA(n_components=n_components, random_state=42)
    scores_arr = pca.fit_transform(X)

    pc_cols = [f"PC{i+1}" for i in range(n_components)]

    scores = pd.DataFrame(scores_arr, index=df.index, columns=pc_cols)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=df_clean.columns,
        columns=pc_cols,
    )

    var_exp = pca.explained_variance_ratio_ * 100
    for i, v in enumerate(var_exp):
        print(f"PC{i+1}: {v:.1f}% variance explained")
    print(f"Total ({n_components} PCs): {var_exp.sum():.1f}%")

    return scores, loadings, pca

def clinical_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a one-column summary DataFrame of key cohort statistics.

    Parameters
    ----------
    df : clinical dataframe (Nivolumab-only or full).

    Returns
    -------
    pd.DataFrame  — index = statistic name, column = 'Value'.
    """
    def _pct(mask):
        return f"{mask.sum()} ({mask.sum()/mask.notna().sum()*100:.1f}%)"

    summary = {
        "Total patients":               len(df),
        "Cohorts":                      ", ".join(sorted(df["Cohort"].dropna().unique())),
        "Treatment arm(s)":             ", ".join(sorted(df["Arm"].dropna().unique())),
        "Median age (years)":           f"{df['Age'].median():.0f}  (IQR {df['Age'].quantile(.25):.0f}–{df['Age'].quantile(.75):.0f})",
        "Male":                         _pct(df["Sex"] == "Male"),
        "MSKCC Favorable":              _pct(df["MSKCC"] == "FAVORABLE"),
        "MSKCC Intermediate":           _pct(df["MSKCC"] == "INTERMEDIATE"),
        "MSKCC Poor":                   _pct(df["MSKCC"] == "POOR"),
        "ORR — CRPR/CR/PR":             df["ORR"].isin(["CRPR", "CR", "PR"]).sum(),
        "ORR — SD":                     (df["ORR"] == "SD").sum(),
        "ORR — PD":                     (df["ORR"] == "PD").sum(),
        "Clinical Benefit (CB)":        (df["Benefit"] == "CB").sum(),
        "No Clinical Benefit (NCB)":    (df["Benefit"] == "NCB").sum(),
        "Indeterminate (ICB)":          (df["Benefit"] == "ICB").sum(),
        "Median PFS (months)":          f"{df['PFS'].median():.1f}",    # PFS stored in months
        "Median OS (months)":           f"{df['OS'].median():.1f}",     # OS stored in months
        "With WES/TMB data":            df["TMB_Counts"].notna().sum(),
        "With CD8 IF data":             df["TM_CD8_Density"].notna().sum(),
        "With RNA/signature data":      df["Angio"].notna().sum() if "Angio" in df.columns else "N/A",
    }
    return pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])


# ─────────────────────────────────────────────────────────────────────────────
# COHORT CHARACTERISATION TABLE  (Table 1)
# ─────────────────────────────────────────────────────────────────────────────



def cohort_characterisation_table(
    df: pd.DataFrame,
    features: list[str],
    group_col: str | None = None,
    continuous_threshold: int = 10,
    p_value_correction: bool = False,
) -> pd.DataFrame:
    """
    Build a publication-ready Table 1 (cohort characterisation table).

    Auto-detects each feature as continuous or categorical based on dtype
    and unique-value count.  When ``group_col`` is provided the table is
    stratified by that column and a statistical test p-value is added.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level clinical dataframe.
    features : list[str]
        Column names to include in the table (order is preserved).
    group_col : str or None
        Column to stratify by (e.g. "ORR_binary", "Benefit", "Arm").
        Pass None for an un-stratified overall-cohort table.
    continuous_threshold : int
        Columns with ≥ this many unique non-null values AND numeric dtype
        are treated as continuous; anything else is categorical.
    p_value_correction : bool
        If True, apply Benjamini-Hochberg FDR correction across all p-values.

    Returns
    -------
    pd.DataFrame
        Multi-row Table 1 with these columns (stratified example)::

            Feature | Category | Overall | Group A | Group B | P-value

        Continuous rows show ``median (IQR)`` with a Mann-Whitney p-value.
        Categorical rows show one sub-row per category with ``n (%)`` and a
        Fisher's-exact (2×2) or Chi-squared (larger) p-value at the first
        sub-row.

    Examples
    --------
    >>> tbl = cohort_characterisation_table(
    ...     df_nivo,
    ...     features=["Age", "Sex", "MSKCC", "TMB_Counts", "ORR"],
    ...     group_col="Benefit",
    ... )
    >>> tbl
    """
    from scipy import stats

    # ── helpers ──────────────────────────────────────────────────────────────

    def _fmt_median_iqr(series: pd.Series) -> str:
        s = series.dropna()
        if len(s) == 0:
            return "—"
        med  = s.median()
        q1   = s.quantile(0.25)
        q3   = s.quantile(0.75)
        return f"{med:.1f} ({q1:.1f}–{q3:.1f})"

    def _fmt_n_pct(n: int, denom: int) -> str:
        if denom == 0:
            return "—"
        return f"{n} ({n/denom*100:.1f}%)"

    def _p_fmt(p: float | None) -> str:
        if p is None or np.isnan(p):
            return "—"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def _is_continuous(col: str) -> bool:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            return False
        return df[col].nunique(dropna=True) >= continuous_threshold

    def _mannwhitney_p(col: str) -> float | None:
        """Mann-Whitney U between the first two groups."""
        grps = [g[col].dropna().values for _, g in df.groupby(group_col)]
        if len(grps) < 2 or any(len(g) == 0 for g in grps[:2]):
            return None
        if len(grps) == 2:
            _, p = stats.mannwhitneyu(grps[0], grps[1], alternative="two-sided")
        else:
            # Kruskal-Wallis for > 2 groups
            _, p = stats.kruskal(*grps)
        return p

    def _cat_p(col: str, cat: str | None) -> float | None:
        """
        Chi-squared or Fisher's exact for a categorical column.
        Returns p only on the first category row (cat=None means 'compute now').
        """
        ct = pd.crosstab(df[col], df[group_col])
        if ct.shape == (2, 2):
            _, p, _, _ = stats.chi2_contingency(ct, correction=False)
            # Use Fisher for small cells
            if ct.values.min() < 5:
                _, p = stats.fisher_exact(ct.values)
        else:
            _, p, _, _ = stats.chi2_contingency(ct)
        return p

    # ── determine groups ──────────────────────────────────────────────────────
    groups: list = []
    if group_col is not None:
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in dataframe.")
        groups = sorted(df[group_col].dropna().unique().tolist())

    # ── build rows ────────────────────────────────────────────────────────────
    records    = []
    p_raw      = []    # collect for optional FDR correction

    for col in features:
        if col not in df.columns:
            print(f"  [WARNING] Feature '{col}' not found — skipping.")
            continue

        n_miss   = df[col].isna().sum()
        n_valid  = df[col].notna().sum()
        miss_str = f" [{n_miss} missing]" if n_miss > 0 else ""

        if _is_continuous(col):
            # ── continuous ─────────────────────────────────────────────────
            overall = _fmt_median_iqr(df[col])
            row: dict = {
                "Feature":   f"{col}{miss_str}",
                "Category":  "Median (IQR)",
                "Overall":   overall,
            }
            for grp in groups:
                row[str(grp)] = _fmt_median_iqr(df.loc[df[group_col] == grp, col])

            p = _mannwhitney_p(col) if group_col else None
            row["P-value"] = _p_fmt(p)
            p_raw.append(p)
            records.append(row)

        else:
            # ── categorical ────────────────────────────────────────────────
            categories = sorted(df[col].dropna().astype(str).unique().tolist())

            # Compute p-value once (for the first sub-row)
            p = _cat_p(col, None) if (group_col and len(categories) > 0) else None
            p_raw.append(p)

            for i, cat in enumerate(categories):
                mask_overall = df[col].astype(str) == cat
                row = {
                    "Feature":  f"{col}{miss_str}" if i == 0 else "",
                    "Category": str(cat),
                    "Overall":  _fmt_n_pct(mask_overall.sum(), n_valid),
                }
                for grp in groups:
                    mask_grp    = (df[group_col] == grp) & mask_overall
                    denom_grp   = (df[group_col] == grp) & df[col].notna()
                    row[str(grp)] = _fmt_n_pct(mask_grp.sum(), denom_grp.sum())

                # p-value only on the first category row
                row["P-value"] = _p_fmt(p) if i == 0 else ""
                records.append(row)

    # ── build DataFrame ───────────────────────────────────────────────────────
    cols = ["Feature", "Category", "Overall"]
    cols += [str(g) for g in groups]
    if group_col:
        cols += ["P-value"]

    table = pd.DataFrame(records, columns=cols)

    # ── optional FDR correction ───────────────────────────────────────────────
    if p_value_correction and group_col:
        valid_p = [(i, p) for i, p in enumerate(p_raw) if p is not None and not np.isnan(p)]
        if valid_p:
            idxs, pvals = zip(*valid_p)
            from statsmodels.stats.multitest import multipletests
            _, corrected, _, _ = multipletests(pvals, method="fdr_bh")
            # Map back — each feature occupies potentially multiple rows;
            # we stored one p per feature, overwrite just the first row per feature
            feat_row_start = 0
            feat_idx = 0
            for row_i, rec in enumerate(records):
                if rec["Feature"] != "":   # first row of a feature
                    if feat_idx < len(idxs) and idxs[feat_idx] == feat_idx:
                        table.at[row_i, "P-value"] = _p_fmt(corrected[feat_idx])
                    feat_idx += 1

    # add N to group column headers  (single-line, no \n)
    if group_col:
        rename = {}
        for grp in groups:
            n_grp = (df[group_col] == grp).sum()
            rename[str(grp)] = f"{grp} (N={n_grp})"
        rename["Overall"] = f"Overall (N={len(df)})"
        table = table.rename(columns=rename)

    return table.reset_index(drop=True)
