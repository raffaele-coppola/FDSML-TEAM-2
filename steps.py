import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.stats import ttest_ind

class VifSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: int = 10, verbose: bool = False):
        self.threshold = threshold
        self.selected_features_ = None
        self.verbose = verbose
        self.custom_step = True
        self.feature_names_in_ = None


    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            data = pd.DataFrame(X)
        else:
            data = X.copy()

        self.feature_names_in_ = np.array(data.columns, dtype=object)

        features: list[str] = [feature for feature in
            data.select_dtypes(include="number").columns.to_list()
            if feature != "target"]

        multicollinear_features = []

        while True:
            current_data = data[features]
            current_data = sm.add_constant(current_data)

            vif_df = pd.DataFrame()
            vif_df["feature"] = current_data.columns.to_list()

            vif_df["vif"] = [variance_inflation_factor(current_data.values, column_index)
                             for column_index in range(current_data.shape[1])]

            vif_df = vif_df[vif_df["feature"] != "const"]

            max_vif_idx = vif_df["vif"].idxmax()
            max_vif = float(vif_df.loc[max_vif_idx, "vif"]) #type: ignore

            if max_vif <= self.threshold:
                break

            feature_to_drop: str = vif_df.loc[max_vif_idx, "feature"] #type: ignore

            if self.verbose:
                print(f"Deleting feature: {feature_to_drop} - VIF: {max_vif:.2f}")

            multicollinear_features.append(feature_to_drop)
            features.remove(feature_to_drop)

        self.selected_features_ = features

        if self.verbose:
            print(f"Selected features: {self.selected_features_}")

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            data = pd.DataFrame(X)
        else:
            data = X.copy()


        return data[self.selected_features_]

    def get_feature_names_out(self):
        return np.array(self.selected_features_, dtype=object)


class WelchSelector(BaseEstimator, TransformerMixin):
    """
    Seleziona le top k feature in base al t-test di Welch.
    Filtra le feature con p-value <= alpha e le ordina per valore assoluto del t-statistic.
    """
    def __init__(self, percentage: float = 0.5, alpha: float = 0.05, verbose: bool = False):
        self.selected_features_ = None
        self.percentage = percentage
        self.alpha = alpha
        self.verbose = verbose
        self.custom_step = True
        self.feature_names_in_ = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.array(X.columns, dtype=object)

        y = pd.Series(np.asarray(y), index=X.index)

        cancer_patients = X[y == 1]
        cancer_free_patients = X[y == 0]

        ttest_results = []

        for col in X.columns:
            malignant_data = cancer_patients[col]
            benign_data = cancer_free_patients[col]

            t_stat, p_val = ttest_ind(malignant_data, benign_data, equal_var=False)

            ttest_results.append({
                "feature": col,
                "t_statistic": abs(t_stat),
                "p_value": p_val,
                "h_0_refused": p_val <= self.alpha
            })

        ttest_df = pd.DataFrame(ttest_results)

        valid_features = ttest_df.query("h_0_refused == True")

        sorted_features = valid_features.sort_values(by="t_statistic", ascending=False)

        # si scelgono le top k feature in base alla percentuale definita
        k = max(1, int(self.percentage * len(sorted_features)))

        self.selected_features_ = sorted_features["feature"].head(k).tolist()
        if self.verbose:
            print(f"Selected features: {self.selected_features_}")

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]

        X_df = pd.DataFrame(X)
        return X_df[self.selected_features_]

    def get_feature_names_out(self):
        return np.array(self.selected_features_, dtype=object)


preprocessing_pipeline = Pipeline(steps=[
    ("scaler", RobustScaler().set_output(transform="pandas")),
    ("vif", VifSelector(threshold=10)),
    ("welch", WelchSelector()),
    ("smote", SMOTE(random_state=42, sampling_strategy="auto"))
])