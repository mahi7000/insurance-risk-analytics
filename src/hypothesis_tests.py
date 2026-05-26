import pandas as pd
from scipy import stats

def test_numerical_margin(group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05) -> dict:
    """Performs an independent two-sample t-test (Welch's t-test) for comparison of premium margins."""
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy='omit')
    return {
        "test_type": "Two-Sample T-Test (Welch)",
        "statistic": float(t_stat),
        "p_value": float(p_val),
        "reject_h0": bool(p_val < alpha)
    }

def test_categorical_frequency(df: pd.DataFrame, group_col: str, action_col: str, alpha: float = 0.05) -> dict:
    """Performs a Chi-Square test of independence to assess variation in claim frequency."""
    contingency_table = pd.crosstab(df[group_col], df[action_col])
    chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency_table)
    return {
        "test_type": "Chi-Square Test of Independence",
        "statistic": float(chi2_stat),
        "p_value": float(p_val),
        "reject_h0": bool(p_val < alpha)
    }