import time
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import cast, Literal, Any

import pandas as pd
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.metrics import (recall_score, precision_score,
                             accuracy_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, StackingClassifier
from imblearn.pipeline import Pipeline

import config
import utils


class NestedCV:
    """ Gestisce l'intero flusso di Nested Cross-Validation per modelli singoli ed ensemble.

        Implementa una validazione nidificata a due livelli: il loop interno ottimizza
        gli iperparametri (tramite GridSearchCV o RandomizedSearchCV), mentre il loop
        esterno valuta la capacità di generalizzazione del modello.

        Args:
            X (pd.DataFrame): Dataset di input privo del target.
            y (pd.Series): Vettore dei target di classificazione.
            n_outer (int, optional): Numero di fold per la cross-validation esterna. Default a 5.
            n_inner (int, optional): Numero di fold per la cross-validation interna. Default a 5.
            n_random_iter (int, optional): Numero di iterazioni per RandomizedSearchCV. Default a 60.
            scoring (str, optional): Metrica di valutazione primaria per la scelta del miglior modello. Default a 'recall'.
            save_dir (Path | None, optional): Percorso della directory per il salvataggio dei report. Se None, usa `config.DATA_DIR`. Default a None.
    """
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 n_outer: int = 5,
                 n_inner: int = 5,
                 n_random_iter: int = 60,
                 scoring: str = 'recall',
                 save_dir: Path | None = None):


        self.X = X
        self.y = y

        self.n_outer = n_outer
        self.n_inner = n_inner
        self.random_iter = n_random_iter

        self.scoring = scoring

        self.outer_cv = StratifiedKFold(n_splits=self.n_outer, shuffle=True, random_state=config.RANDOM_STATE)
        self.inner_cv = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=config.RANDOM_STATE)

        self.logger = utils.setup_logger("nested-cv-logger")

        self.prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = save_dir or config.DATA_DIR

        # key: nome modello, value: dict con key: outer fold, value: dict con key: search type e value best params
        self.best_model_params: dict[str, dict[int, dict[str, dict]]] = {}

        # key: nome modello, value: dict con key: outer fold, value: dict con key: search type e value tempo di fitting
        self.best_model_fit_times: dict[str, dict[int, dict[str, float]]] = {}

    @staticmethod
    def get_grid_total_combination_number(grid_params: dict | list) -> int:
        """ Calcola il numero totale di combinazioni generate da una griglia di parametri.

            Supporta sia una singola griglia rappresentata come dizionario di liste,
            sia un elenco di più griglie distinte.

            Args:
                grid_params (dict | list): Iperparametri per la grid search. Può essere
                    un dizionario `{parametro: [valori]}` o una lista di dizionari.

            Returns:
                int: Numero totale di combinazioni uniche da esplorare. Restituisce 0 se
                    l'input è vuoto o di tipo non supportato.
        """
        if not grid_params:
            return 0

        if isinstance(grid_params, dict):
            counter = 1
            for param, values in grid_params.items():
                counter *= len(values)
            return counter

        elif isinstance(grid_params, list):
            total = 0
            for combination in grid_params:
                counter = 1
                for param, values in combination.items():
                    counter *= len(values)

                total += counter

            return total

        return 0

    @staticmethod
    def _get_selected_features(fitted_pipeline: Pipeline) -> dict[str, Any]:
        """ Estrae i nomi delle feature selezionate dai passaggi di una pipeline addestrata.

            Ispeziona i selettori custom a monte e quelli annidati all'interno di
            modelli di tipo ensemble (VotingClassifier o StackingClassifier).

            Args:
                fitted_pipeline (Pipeline): Istanza di Pipeline di scikit-learn già addestrata.

            Returns:
                dict[str, Any]: Dizionario che mappa i nomi degli step di selezione
                    alle liste di feature mantenute.
        """

        features = {}

        for step_name, step in fitted_pipeline.named_steps.items():
            if hasattr(step, "get_feature_names_out") and hasattr(step, "custom_step"):
                names = step.get_feature_names_out()
                features[step_name] = names.tolist() if hasattr(names, "tolist") else list(names)

        model_step = fitted_pipeline.named_steps.get("model")
        if isinstance(model_step, (VotingClassifier, StackingClassifier)) and hasattr(model_step, "estimators_"):
            ensemble_features = {}
            for (name, _), fitted_estimator in zip(model_step.estimators, model_step.estimators_):
                if isinstance(fitted_estimator, Pipeline) and "welch" in fitted_estimator.named_steps:
                    welch_step = fitted_estimator.named_steps["welch"]
                    names = welch_step.get_feature_names_out()
                    ensemble_features[name] = names.tolist() if hasattr(names, "tolist") else list(names)

            if ensemble_features:
                features["welch"] = ensemble_features

        return features

    @staticmethod
    def _clean_best_params(best_params: dict) -> dict:
        """Converte i tipi numerici NumPy di un dizionario nei tipi nativi Python.

            Trasforma i valori di tipo `np.floating` in `float` standard e
            `np.integer` in `int`.

            Args:
                best_params (dict): Dizionario contenente i migliori parametri trovati.

            Returns:
                dict: Nuovo dizionario con gli stessi parametri ma con tipi numerici nativi.
        """
        clean_params = {}

        for key, value in best_params.items():
            if isinstance(value, np.floating):
                clean_params[key] = float(value)
            elif isinstance(value, np.integer):
                clean_params[key] = int(value)
            else:
                clean_params[key] = value

        return clean_params

    @staticmethod
    def _get_y_proba(estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame) -> tuple[np.ndarray, float, CalibratedClassifierCV | None]:
        """ Calcola le probabilità della classe positiva misurando il tempo di inferenza.

            Se lo stimatore non implementa nativamente `predict_proba`, viene addestrato
            un `CalibratedClassifierCV` sui dati di training per calibrare le stime.

            Args:
                estimator (Pipeline): Pipeline o stimatore da utilizzare per la predizione.
                X_train (pd.DataFrame): Dati di feature per l'addestramento dell'eventuale calibratore.
                y_train (pd.Series): Target per l'addestramento dell'eventuale calibratore.
                X_test (pd.DataFrame): Dati di feature di test su cui calcolare le probabilità.

            Returns:
                tuple[np.ndarray, float, CalibratedClassifierCV | None]: Tupla contenente:
                    - Array delle probabilità predette per la classe positiva (1).
                    - Tempo impiegato (in secondi) per eseguire la predizione.
                    - Istanza del classificatore calibrato se addestrato, altrimenti None.
            """
        if hasattr(estimator, "predict_proba"):
            start_time = time.perf_counter()
            y_proba = estimator.predict_proba(X_test)[:, 1]
            elapsed = time.perf_counter() - start_time
            return y_proba, elapsed, None

        # per i modelli senza predict proba si usa questo wrapper
        calibrated = CalibratedClassifierCV(
            estimator=clone(estimator),
            cv=5,
            ensemble=False
        )
        calibrated.fit(X_train, y_train)

        start_time = time.perf_counter()
        y_proba = calibrated.predict_proba(X_test)[:, 1]
        elapsed = time.perf_counter() - start_time
        return y_proba, elapsed, calibrated

    @staticmethod
    def _results_to_dataframe(results: list[dict]) -> pd.DataFrame:
        """Converte la lista dei risultati della nested cross-validation in un DataFrame.

            Estrae le metriche di performance, i metadati dei fold e le feature selezionate
            per ciascuna strategia di ricerca ("grid" e "random") all'interno di ogni outer fold.

            Args:
                results (list[dict]): Lista in cui il primo elemento specifica il modello
                    e i successivi contengono i risultati dettagliati per ogni fold.

            Returns:
                pd.DataFrame: DataFrame contenente una riga per ciascuna combinazione di
                    outer fold e strategia di ricerca, con tutte le metriche e configurazioni.
        """
        model_name = results[0]["model"]
        rows = []
        for fold_result in results[1:]:
            for search_type in ["grid", "random"]:
                metrics = fold_result[search_type]
                selected_feature_keys = [key for key in metrics.keys()
                                         if str(key).endswith("_selected_features")]
                rows.append({
                    "model": model_name,
                    "best_estimator": metrics["best_estimator"],
                    "outer_fold": fold_result["outer_fold"],
                    "y_true": fold_result["y_true"],
                    "y_pred": metrics["y_pred"],
                    "y_proba": metrics["y_proba"],
                    "search_type": search_type,
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "inner_best_score": metrics["inner_best_score"],
                    "inner_mean_validation_score": metrics["inner_mean_validation_score"],
                    "inner_std_validation_score": metrics["inner_std_validation_score"],
                    "fit_time_seconds": metrics["fit_time_seconds"],
                    "inference_time_seconds": metrics["inference_time_seconds"],
                    "inference_proba_time_seconds": metrics["inference_proba_time_seconds"],
                    "best_params": metrics["best_params"],
                    "train_indexes": fold_result["train_indexes"],
                    "test_indexes": fold_result["test_indexes"],
                    **{key: metrics[key] for key in selected_feature_keys}
                })

        df = pd.DataFrame(rows)
        print(df.columns.to_list())
        return df

    def _add_params(self, model_params: dict, params_to_add: dict):
        """ Aggiunge parametri supplementari alle configurazioni di ricerca del modello.

            Esegue una copia profonda dei parametri esistenti e vi unisce i nuovi parametri
            per entrambe le strategie ("grid" e "random"), gestendo sia dizionari singoli
            sia liste di dizionari.

            Args:
                model_params (dict): Parametri correnti del modello strutturati per tipo di ricerca.
                params_to_add (dict): Nuovi parametri da aggiungere, contenenti obbligatoriamente
                    le chiavi "grid" e "random".

            Returns:
                dict: Copia aggiornata dei parametri del modello con le nuove opzioni incorporate,
                    oppure i parametri originali invariati se la validazione fallisce.
        """
        if not model_params:
            return model_params

        params = deepcopy(model_params)

        if ("random" not in params_to_add) or ("grid" not in params_to_add):
            self.logger.warning(
                "Added params do not have expected syntax! Using model params as default"
            )
            return params

        for search_type in ["grid", "random"]:
            search_model_params = params[search_type]

            if isinstance(search_model_params, list):
                for param in search_model_params:
                    param.update(params_to_add[search_type])
            else:
                search_model_params.update(params_to_add[search_type])

        return params

    def _save_results(self, results_df: pd.DataFrame,
                      filename: str | None = None,
                      output_format: Literal["csv", "parquet", "pickle"] = "pickle"):
        """ Salva il DataFrame dei risultati sul disco nel formato specificato.

            Args:
                results_df (pd.DataFrame): DataFrame contenente i risultati da salvare.
                filename (str | None, optional): Nome del file di destinazione. Se non specificato,
                    viene generato automaticamente in base ai metadati del DataFrame. Default a None.
                output_format (Literal["csv", "parquet", "pickle"], optional): Formato del file
                    di output ("csv", "parquet" o "pickle"). Default a "pickle".
        """
        method_name = f"to_{output_format}"

        if not hasattr(results_df, method_name):
            self.logger.warning(
                f"Could not save dataframe: unsupported output format '{output_format}'."
            )
            return

        save_method = getattr(results_df, method_name)

        output_format = output_format.replace("pickle", "pkl")

        # modello singolo
        if len(results_df) == (len(results_df['search_type'].unique()) * len(results_df['outer_fold'].unique())):
            model_name = results_df["model"].iloc[0]
            filename = f"{self.prefix}_{model_name}_nested_cv.{output_format}"

        if not filename:
            filename = config.NESTED_RESULTS_FILENAME

        output_path = self.save_dir / filename

        save_method(output_path)

    def _save_best_model_data(self, model_results: pd.DataFrame):
        """Aggiorna i dizionari interni con i migliori parametri e i tempi di addestramento.

            Args:
                model_results (pd.DataFrame): DataFrame contenente i risultati del modello,
                    inclusi nome, outer fold, strategia di ricerca, parametri ottimali e tempi di fit.
        """
        model_name = model_results["model"].iloc[0]
        if model_name not in self.best_model_params:
            self.best_model_params[model_name] = {}
            self.best_model_fit_times[model_name] = {}

        for _, row in model_results.iterrows():
            outer_fold = int(row["outer_fold"])
            search_type = row["search_type"]

            if outer_fold not in self.best_model_params[model_name]:
                self.best_model_params[model_name][outer_fold] = {"grid": {}, "random": {}}
                self.best_model_fit_times[model_name][outer_fold] = {"grid": 0.0, "random": 0.0}

            self.best_model_params[model_name][outer_fold][search_type] = row["best_params"]
            self.best_model_fit_times[model_name][outer_fold][search_type] = row["fit_time_seconds"]

    def _compute_ensemble_fit_time(self, model: StackingClassifier | VotingClassifier, outer_fold: int,
                                   search_type: str,
                                   elapsed_time: float = 0.0) -> float:
        """ Stima il tempo totale di addestramento per un modello ensemble.

            Recupera il tempo massimo di fit tra i modelli base (assumendo esecuzione parallela)
            e, nel caso di uno StackingClassifier, vi aggiunge il tempo impiegato per
            l'addestramento del meta-modello.

            Args:
                model (StackingClassifier | VotingClassifier): Modello ensemble di cui
                    calcolare il tempo di addestramento.
                outer_fold (int): Indice dell'outer fold corrente.
                search_type (str): Strategia di ricerca utilizzata ("grid" o "random").
                elapsed_time (float, optional): Tempo di fit specifico dell'ensemble o del
                    meta-modello (in secondi). Default a 0.0.

            Returns:
                float: Tempo stimato complessivo di fit in secondi.
        """

        max_base_time = 0.0

        for name, estimator in model.estimators:
            if isinstance(estimator, Pipeline) and "clf" in estimator.named_steps:
                est_clf = estimator.named_steps["clf"]
            else:
                est_clf = estimator

            est_class = est_clf.__class__.__name__

            if est_class in self.best_model_fit_times:
                model_name = est_class
            elif name in self.best_model_fit_times:
                model_name = name
            else:
                model_name = None

            if model_name:
                base_time = self.best_model_fit_times[model_name].get(outer_fold, {}).get(search_type, 0.0)
                if base_time > max_base_time:
                    max_base_time = base_time

        if isinstance(model, VotingClassifier):
            final_fit_time = max_base_time
        else:
            final_fit_time = max_base_time + elapsed_time

        return final_fit_time


    def _build_search_result(self,
                             best_estimator: Pipeline, best_params: dict,
                             inner_best_score: float, inner_mean: float, inner_std: float,
                             fit_time: float, X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series):
        """ Valuta il miglior estimator su un fold di test e registra metriche e metadati.

            Calcola le metriche di classificazione, misura i tempi di inferenza (predizione
            e probabilità), estrae le feature selezionate e restituisce l'intero riepilogo
            in un dizionario.

            Args:
                best_estimator (Pipeline): Pipeline ottimizzata selezionata dalla ricerca interna.
                best_params (dict): Migliori iperparametri trovati durante la ricerca.
                inner_best_score (float): Miglior punteggio ottenuto nell'inner loop.
                inner_mean (float): Media dei punteggi di validazione dell'inner loop.
                inner_std (float): Deviazione standard dei punteggi di validazione dell'inner loop.
                fit_time (float): Tempo totale impiegato per l'addestramento (in secondi).
                X_train (pd.DataFrame): Dati di addestramento per l'eventuale calibrazione.
                y_train (pd.Series): Target di addestramento per l'eventuale calibrazione.
                X_test (pd.DataFrame): Dati del set di test.
                y_test (pd.Series): Target reale del set di test.
        """

        pred_start_time = time.perf_counter()
        predictions = best_estimator.predict(X_test)
        pred_elapsed = time.perf_counter() - pred_start_time

        y_proba, proba_elapsed_time, calibrated_estimator = self._get_y_proba(best_estimator,
                                                                              X_train, y_train,
                                                                              X_test)

        selected_features = self._get_selected_features(best_estimator)

        return {
            "best_estimator": calibrated_estimator or best_estimator,

            "recall": round(float(recall_score(y_test, predictions)), 3),
            "precision": round(float(precision_score(y_test, predictions)), 3),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
            "f1": round(float(f1_score(y_test, predictions)), 3),

            "best_params": best_params,

            "inner_best_score": round(inner_best_score, 3) if not np.isnan(inner_best_score) else np.nan,
            "inner_mean_validation_score": round(inner_mean, 3) if not np.isnan(inner_mean) else np.nan,
            "inner_std_validation_score": round(inner_std, 3) if not np.isnan(inner_std) else np.nan,

            "fit_time_seconds": round(fit_time, 2),
            "inference_time_seconds": round(pred_elapsed, 4),
            "inference_proba_time_seconds": round(proba_elapsed_time, 4),

            "y_pred": predictions,
            "y_proba": y_proba,

            **{f"{k}_selected_features": v for k, v in selected_features.items()},
        }

    def _inject_ensemble_params(
            self,
            ensemble: StackingClassifier | VotingClassifier,
            outer_fold: int,
            search_type: str
    ) -> StackingClassifier | VotingClassifier:
        """Inietta i migliori iperparametri nei modelli base di un classificatore ensemble.

            Clona l'ensemble e, per ogni stimatore base, applica i parametri ottimali trovati
            nel rispettivo outer fold e tipo di ricerca, rimappando i prefissi se necessario.
            Nel caso di un VotingClassifier, avvolge automaticamente con CalibratedClassifierCV
            gli stimatori sprovvisti del metodo `predict_proba`.

            Args:
                ensemble (StackingClassifier | VotingClassifier): Modello ensemble da configurare.
                outer_fold (int): Indice dell'outer fold corrente da cui recuperare i parametri.
                search_type (str): Strategia di ricerca utilizzata ("grid" o "random").

            Returns:
                StackingClassifier | VotingClassifier: Nuova istanza dell'ensemble con stimatori base
                    configurati e calibrati.
        """

        ensemble_clone = clone(ensemble)
        if not isinstance(ensemble_clone, (StackingClassifier, VotingClassifier)):
            raise RuntimeError()

        configured = []

        for name, estimator in ensemble_clone.estimators:
            est_clone = cast(BaseEstimator, clone(estimator))

            if isinstance(est_clone, Pipeline) and "clf" in est_clone.named_steps:
                est_clf = est_clone.named_steps["clf"]
                est_class = est_clf.__class__.__name__
            else:
                est_class = est_clone.__class__.__name__

            if est_class in self.best_model_params:
                model_name = est_class
            elif name in self.best_model_params:
                model_name = name
            else:
                model_name = None

            if model_name:
                fold_params = self.best_model_params[model_name].get(outer_fold, {}).get(search_type, {})
                branch_params = {}
                for k, v in fold_params.items():
                    if k.startswith("model__"):
                        if isinstance(est_clone, Pipeline) and "clf" in est_clone.named_steps:
                            branch_params[k.replace("model__", "clf__")] = v
                        else:
                            branch_params[k.replace("model__", "")] = v
                    elif k.startswith("welch__"):
                        if isinstance(est_clone, Pipeline) and "welch" in est_clone.named_steps:
                            branch_params[k] = v

                if branch_params:
                    est_clone.set_params(**branch_params)

            # calibrazione per il voting classifier
            if isinstance(ensemble_clone, VotingClassifier):
                clf_to_check = est_clone.named_steps["clf"] if isinstance(est_clone,
                                                                          Pipeline) and "clf" in est_clone.named_steps else est_clone
                if not hasattr(clf_to_check, "predict_proba"):
                    calibrated_clf = CalibratedClassifierCV(
                        estimator=clf_to_check,
                        cv=self.inner_cv,
                        ensemble=False
                    )
                    if isinstance(est_clone, Pipeline) and "clf" in est_clone.named_steps:
                        est_clone.set_params(**{"clf": calibrated_clf})
                    else:
                        est_clone = calibrated_clf

            configured.append((name, est_clone))

        ensemble_clone.estimators = configured

        if hasattr(ensemble_clone, "cv"):
            ensemble_clone.cv = self.inner_cv

        return ensemble_clone


    def _run_search(self, pipeline: Pipeline,
                    outer_fold: int, search_type: Literal["grid", "random"], search_params: dict,
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series):

        """ Esegue la ricerca degli iperparametri (o il fit diretto) per un singolo outer fold.

            Gestisce l'iniezione preventiva dei parametri per i modelli ensemble, esegue
            `GridSearchCV` o `RandomizedSearchCV` sulla cross-validation interna (o il fit
            semplice se non sono previsti parametri), calcola i tempi di addestramento effettivi
            e delega il salvataggio delle metriche di valutazione.

            Args:
                pipeline (Pipeline): Pipeline  di base da ottimizzare o addestrare.
                outer_fold (int): Indice dell'outer fold corrente.
                search_type (Literal["grid", "random"]): Strategia di ricerca ("grid" o "random").
                search_params (dict): Spazio dei parametri su cui iterare; se vuoto, esegue un fit singolo.
                X_train (pd.DataFrame): Feature del set di training per la ricerca interna/fit.
                y_train (pd.Series): Target del set di training per la ricerca interna/fit.
                X_test (pd.DataFrame): Feature del set di test per la valutazione finale del fold.
                y_test (pd.Series): Target del set di test per la valutazione finale del fold.
        """

        search_pipeline = cast(Pipeline, clone(pipeline))

        original_model_step = pipeline.named_steps["model"]

        if isinstance(original_model_step, (StackingClassifier, VotingClassifier)):
            configured_ensemble = self._inject_ensemble_params(
                original_model_step, outer_fold=outer_fold, search_type=search_type
            )
            search_pipeline.set_params(**{"model": configured_ensemble})

        # se ci sono iperparametri di ricerca, avvia la grid/random search
        if search_params:
            if search_type == "grid":
                search = GridSearchCV(
                    estimator=search_pipeline, param_grid=search_params, scoring=self.scoring,
                    cv=self.inner_cv, n_jobs=-1, refit=True, error_score="raise"
                )
            else:
                search = RandomizedSearchCV(
                    estimator=search_pipeline, param_distributions=search_params, scoring=self.scoring,
                    n_iter=self.random_iter, cv=self.inner_cv, n_jobs=-1, refit=True, error_score="raise"
                )

            search_start = time.perf_counter()
            search.fit(X_train, y_train)
            elapsed_time = time.perf_counter() - search_start

            best_estimator = search.best_estimator_
            best_params = self._clean_best_params(search.best_params_)
            inner_best = float(search.best_score_)
            inner_mean = search.cv_results_['mean_test_score'][search.best_index_]
            inner_std = search.cv_results_['std_test_score'][search.best_index_]

        # se non ci sono iperparametri, effettua il canonico fit singolo (voting classifier)
        else:
            search_start = time.perf_counter()
            search_pipeline.fit(X_train, y_train)
            elapsed_time = time.perf_counter() - search_start

            best_estimator = search_pipeline
            best_params = {}
            inner_best, inner_mean, inner_std = np.nan, np.nan, np.nan

        if not isinstance(original_model_step, (StackingClassifier, VotingClassifier)):
            final_fit_time = elapsed_time
        else:
            base_params = {}
            for name, estimator in original_model_step.estimators:
                if isinstance(estimator, Pipeline) and "clf" in estimator.named_steps:
                    est_clf = estimator.named_steps["clf"]
                else:
                    est_clf = estimator

                est_class = est_clf.__class__.__name__

                if est_class in self.best_model_params:
                    model_name = est_class
                elif name in self.best_model_params:
                    model_name = name
                else:
                    model_name = None

                if model_name:
                    fold_params = self.best_model_params[model_name].get(outer_fold, {}).get(search_type, {})
                    for k, v in fold_params.items():
                        clean_k = k.replace("model__", "clf__")
                        base_params[f"{name}__{clean_k}"] = v

            best_params = {**base_params, **best_params}

            final_fit_time = self._compute_ensemble_fit_time(model=original_model_step, outer_fold=outer_fold,
                                                             search_type=search_type, elapsed_time=elapsed_time)

        return self._build_search_result(
            best_estimator=best_estimator, best_params=best_params,
            inner_best_score=inner_best, inner_mean=inner_mean, inner_std=inner_std,
            fit_time=final_fit_time,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

    def run(self, pipeline: Pipeline, model_params: dict,
                  model = None, params_to_add: dict | None = None,
                  save_results: bool = True, filename: str | None = None,
                  output_format: Literal["csv", "parquet", "pickle"] = "pickle") -> pd.DataFrame:
        """ Esegue la procedura completa di nested cross-validation per un dato modello.

            Per ciascun outer fold valuta sia la ricerca su griglia ("grid") sia quella
            casuale ("random") eseguite tramite inner cross-validation, raccoglie i risultati,
            registra metriche e tempi di esecuzione e opzionalmente salva il DataFrame finale su disco.

            Args:
                pipeline (Pipeline): Pipeline di base scikit-learn da utilizzare nel processo.
                model_params (dict): Spazio dei parametri di ricerca, contenente le chiavi "grid" e "random".
                model (Any, optional): Modello/stimatore da aggiungere come step "model" alla pipeline.
                    Default a None.
                params_to_add (dict | None, optional): Parametri addizionali da iniettare
                    in quelli esistenti prima dell'ottimizzazione. Default a None.
                save_results (bool, optional): Indica se persistere il DataFrame dei risultati
                    su file. Default a True.
                filename (str | None, optional): Nome del file di destinazione per il salvataggio.
                    Se None, viene ricavato automaticamente. Default a None.
                output_format (Literal["csv", "parquet", "pickle"], optional): Formato di salvataggio
                    dei risultati. Default a "pickle".

            Returns:
                pd.DataFrame: DataFrame contenente le predizioni, le probabilità e le metriche
                    di validazione dettagliate per ciascun outer fold e strategia di ricerca.
            """

        results = []

        params = deepcopy(model_params) if not params_to_add else self._add_params(
            model_params=model_params, params_to_add=params_to_add)

        working_pipeline = cast(Pipeline, clone(pipeline))

        if model is not None and "model" in working_pipeline.named_steps:
                raise RuntimeError(f"Passed a model but the pipeline already has one: {pipeline.named_steps['model']}")

        working_pipeline.steps.append(("model", clone(model)))
        model_name = working_pipeline.named_steps["model"].__class__.__name__
        results.append({"model": model_name})

        self.logger.info(f"[{model_name}] Starting nested CV — {self.n_outer} outer fold, "
                    f"{self.n_inner} inner fold, scoring='{self.scoring}'")

        total_combination = self.get_grid_total_combination_number(params.get("grid", {}))
        total_grid_training_number = self.n_outer * self.n_inner * total_combination
        self.logger.info(f"Total training number for grid search: {total_grid_training_number} equivalent to "
                    f"{total_combination} total combination * {self.n_outer} * {self.n_inner}")

        total_random_training_number = self.random_iter * self.n_outer * self.n_inner
        self.logger.info(f"Total training number for random search: {total_random_training_number} equivalent to "
                         f"{self.random_iter} random iterations * {self.n_outer} * {self.n_inner}")

        total_start = time.perf_counter()

        for fold, (train_index, test_index) in enumerate(self.outer_cv.split(self.X, self.y), 1):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            fold_result = {
                "outer_fold": fold,
                "y_true": y_test.to_numpy(),
                "train_indexes": train_index,
                "test_indexes": test_index,
            }

            for search_type in ["grid", "random"]:
                search_params = params.get(search_type, {})

                fold_result[search_type] = self._run_search(
                    pipeline=working_pipeline,
                    outer_fold=fold, search_type=search_type, search_params=search_params,  # type: ignore
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test  # type: ignore
                )

            results.append(fold_result)

            grid_time = fold_result.get("grid", {}).get("fit_time_seconds", 0) #type: ignore
            random_time = fold_result.get("random", {}).get("fit_time_seconds", 0) #type: ignore

            grid_recall = fold_result.get("grid", {}).get("recall", 0) #type: ignore
            random_recall = fold_result.get("random", {}).get("recall", 0) #type: ignore

            self.logger.info(f"[{model_name}] Fold {fold}/{self.n_outer} — "
                             f"grid: recall={grid_recall:.3f} in {grid_time:.1f}s — "
                             f"random: recall={random_recall:.3f} in {random_time:.1f}s")

        total_elapsed = time.perf_counter() - total_start
        self.logger.info(f"[{model_name}] Completed in {total_elapsed:.1f}s "
                         f"(~{total_elapsed / self.n_outer:.1f}s/outer fold)")

        results_df = self._results_to_dataframe(results)
        if save_results:
            self._save_results(results_df, filename=filename, output_format=output_format)

        return results_df

    def _load_saved_results(self, models: dict[str, Any]) -> pd.DataFrame | None:
        """ Carica e valida i risultati pregressi salvati su disco.

            Verifica l'esistenza del file serializzato e si assicura che il DataFrame
            memorizzato corrisponda esattamente alla configurazione corrente in termini
            di modelli previsti, outer fold, strategie di ricerca e numero totale di righe.

            Args:
                models (dict[str, Any]): Dizionario contenente i modelli attesi da confrontare
                    con quelli presenti nei risultati salvati.

            Returns:
                pd.DataFrame | None: Il DataFrame dei risultati validati se compatibili
                    con l'esecuzione corrente, altrimenti None se il file non esiste
                    o la struttura non coincide.
        """
        filepath = self.save_dir / config.NESTED_RESULTS_FILENAME
        if filepath.exists():
            results = pd.read_pickle(filepath)
            saved_models = set(results['model'].unique())
            saved_outer_folds = set(results['outer_fold'].unique())
            saved_search_types = set(results['search_type'].unique())

            expected_models = set(list(models.keys()))
            expected_search_types = {'grid', 'random'}
            expected_outer_folds = set(list(range(1, self.n_outer + 1)))
            expected_df_len = len(expected_models) * len(expected_search_types) * len(expected_outer_folds)

            
            same_models = saved_models == expected_models
            same_outer_folds = saved_outer_folds == expected_outer_folds
            same_search_types = saved_search_types == expected_search_types
            same_len = len(results) == expected_df_len

            if same_models and same_outer_folds and same_search_types and same_len:
                self.logger.info(f"Since force recomputing was false, loading results from {filepath}")
                return results

        return None



    def batch_run(self, pipeline: Pipeline,
                  models: dict[str, Any], model_params: dict[str, dict[str, dict | list]],
                  params_to_add: dict | None = None,
                  save_results: bool = True, filename: str | None = None,
                  output_format: Literal["csv", "parquet", "pickle"] = "pickle",
                  force_recompute: bool = False
                  ) -> pd.DataFrame:
        """ Esegue la nested cross-validation sequenziale su un insieme di modelli singoli ed ensemble.

            Addestra e ottimizza prima i modelli base salvandone i migliori parametri e tempi di fit,
            quindi valuta gli eventuali modelli ensemble (VotingClassifier o StackingClassifier)
            utilizzando una pipeline dedicata priva del passaggio 'welch' a monte e iniettando
            i metadati ricavati dal primo livello.

            Args:
                pipeline (Pipeline): Pipeline scikit-learn di base comune a tutti i modelli.
                models (dict[str, Any]): Mappatura `{nome_modello: istanza_modello}` degli stimatori da testare.
                model_params (dict[str, dict[str, dict | list]]): Spazi dei parametri per ciascun modello,
                    organizzati per nome e strategia di ricerca ("grid" e "random").
                params_to_add (dict | None, optional): Ulteriori parametri globali da aggregare alle ricerche.
                    Eventuali parametri con prefisso "welch__" vengono esclusi per gli ensemble. Default a None.
                save_results (bool, optional): Se True, persiste i risultati intermedi e finali su disco. Default a True.
                filename (str | None, optional): Nome del file per il salvataggio complessivo. Se None,
                    viene generato automaticamente. Default a None.
                output_format (Literal["csv", "parquet", "pickle"], optional): Formato di serializzazione
                    dei file di output. Default a "pickle".
                force_recompute (bool, optional): Se True, forza la rielaborazione dei modelli anche se i risultati
                    esistono già. Default a False.
            Returns:
                pd.DataFrame: DataFrame consolidato contenente le metriche e i metadati di tutti i modelli valutati.
        """
        if not force_recompute:
            saved_results = self._load_saved_results(models=models)
            if isinstance(saved_results, pd.DataFrame):
                return saved_results

        all_results = []
        batch_start = time.perf_counter()

        ensemble_models = {}
        for model_name, model in models.items():
            if isinstance(model, (VotingClassifier, StackingClassifier)):
                ensemble_models[model_name] = model
                continue

            model_param = model_params[model_name]
            model_results = self.run(
                pipeline=pipeline, model=model, model_params=model_param, params_to_add=params_to_add,
                save_results=save_results, output_format=output_format
            )
            self._save_best_model_data(model_results)
            all_results.append(model_results)

        ensemble_pipeline = cast(Pipeline, clone(pipeline))
        ensemble_pipeline.steps = [
            (name, step) for name, step in ensemble_pipeline.steps if name != "welch"
        ]


        for ensemble_model_name, ensemble_model in ensemble_models.items():
            if not self.best_model_params:
                self.logger.warning("No first layer params found for blender. Skipping ensemble training.")
            else:
                ensemble_params = model_params.get(ensemble_model_name, {})
                ensemble_results = self.run(
                    pipeline=ensemble_pipeline,
                    model=ensemble_model,
                    model_params=ensemble_params,
                    save_results=True,
                    output_format=output_format
                )
                all_results.append(ensemble_results)

        batch_elapsed = time.perf_counter() - batch_start
        self.logger.info(f"Batch run completed in {batch_elapsed:.1f}s")

        all_results_df = pd.concat(all_results, axis=0, ignore_index=True)

        if save_results:
            self._save_results(all_results_df, filename=filename, output_format=output_format)

        return all_results_df
