import pandas as pd
from config import DATASET_PATH
from typing import cast
from pathlib import Path
import logging
from datetime import datetime


def _load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
    else:
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        df = cast(pd.DataFrame, data.frame)

    return df

def get_cleaned_dataset() -> pd.DataFrame:
    df = _load_dataset()

    df.rename(columns={"Unnamed: 0": "patient_id"}, inplace=True)
    df.set_index("patient_id", inplace=True)

    df.rename(columns={col: col.replace(" ", "_") for col in df.columns}, inplace=True)

    df['target'] = 1 - df['target']

    target_mapping = {1: "Malignant", 0: "Benign"}
    df['diagnosis'] = df['target'].map(target_mapping)

    return df

def setup_logger(name: str, log_file: Path | str | None = None, timestamp: bool = False) -> logging.Logger:
    """
        Configura e restituisce un'istanza di logging.Logger con output su console e file.

        Se viene fornito un percorso per il file, i log verranno scritti anche su disco,
        creando automaticamente le directory necessarie.

        Args:
            name: Nome identificativo del logger.
            log_file: Percorso opzionale del file di log. Se None, il logger scriverà
                solo sulla console.
            timestamp: Se True, aggiunge un timestamp (giorno-mese-anno_ora-min-sec)
                al nome del file di log per evitare sovrascritture.

        Returns:
            logging.Logger: Un'istanza di Logger configurata con StreamHandler (console)
                ed eventualmente FileHandler.
        """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)

            if timestamp:
                log_path = log_path.with_stem(log_path.stem + "_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def extract_best_params(results_df: pd.DataFrame) -> tuple[dict[str, dict[int, dict[str, dict]]],  dict[str, dict[int, dict[str, float]]]]:
    base_models = ["SVC", "RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier"]

    best_params = {}
    fit_times = {}

    subset = results_df[results_df['model'].isin(base_models)]

    for _, row in subset.iterrows():
        model_name = row["model"]
        outer_fold = row["outer_fold"]
        search_type = row["search_type"]
        best_param = row["best_params"]
        fit_time = row["fit_time_seconds"]

        if model_name not in best_params:
            best_params[model_name] = {}
        if model_name not in fit_times:
            fit_times[model_name] = {}

        if outer_fold not in best_params[model_name]:
            best_params[model_name][outer_fold] = {}
        if outer_fold not in fit_times[model_name]:
            fit_times[model_name][outer_fold] = {}

        if search_type not in best_params[model_name][outer_fold]:
            best_params[model_name][outer_fold][search_type] = {}
        if search_type not in fit_times[model_name][outer_fold]:
            fit_times[model_name][outer_fold][search_type] = {}


        best_params[model_name][outer_fold][search_type] = best_param
        fit_times[model_name][outer_fold][search_type] = fit_time


    return best_params, fit_times