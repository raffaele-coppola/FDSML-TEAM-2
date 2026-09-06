# FDSML Team 2

## Descrizione del progetto

Il progetto implementa una pipeline di Machine Learning per la classificazione del dataset [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

Sono stati valutati diversi modelli di classificazione, sia semplici sia ensemble, utilizzando due strategie di ottimizzazione degli iperparametri, ossia `grid search` e `random search`.

I modelli considerati includono:
- `SVC`;
- `RandomForestClassifier`;
- `KNeighborsClassifier`;
- `LogisticRegression`;
- `VotingClassifier`;
- `StackingClassifier`.

L'intero processo e stato incapsulato all'interno di una Nested Cross-Validation. Questa metodologia permette di separare correttamente:

- la fase di ottimizzazione degli iperparametri, eseguita nell'inner loop;
- la fase di valutazione della robustezza e della capacita di generalizzazione del modello, eseguita nell'outer loop.

In questo modo, le prestazioni finali vengono misurate su dati non utilizzati durante la selezione degli iperparametri, riducendo il rischio di ottenere valutazioni ottimistiche o affette da data leakage.

La pipeline comprende inoltre diversi passaggi di preprocessing, tra cui la standardizzazione dei dati, la rimozione iterativa delle feature multicollineari tramite VIF, la selezione delle feature statisticamente rilevanti tramite il test di Welch e il bilanciamento delle classi tramite SMOTE. Ulteriori dettagli presenti nella documentazione in `doc/report.pdf`.


 
## Struttura del progetto

```text
FDSML-TEAM-2/
├── pyproject.toml
├── uv.lock
├── README.md
├── config.py
├── nested_cv.py
├── steps.py
├── utils.py
├── jupyters/
│   ├── eda.ipynb
│   ├── pipeline.ipynb
│   └── results_analysis.ipynb
├── data/
│   ├── dataset.csv
│   ├── nested_cv_results.pkl
│   └── shap_results.pkl
└── doc/
    └── report.pdf
```

### `config.py`

Contiene le principali costanti di configurazione del progetto, tra cui i modelli di classificazione valutati e gli spazi degli iperparametri esplorati da Grid Search e Random Search.



### `steps.py`

Contiene gli step personalizzati di preprocessing utilizzati nella pipeline: `RobustScaler` -> `VifSelector` -> `WelchSelector` -> `WelchSelector`.


### `utils.py`

Contiene funzioni di utility generale.

### `nested_cv.py`

Contiene la logica principale per l'esecuzione della Nested Cross-Validation.

La classe `NestedCV` gestisce:

- la suddivisione del dataset nei fold esterni e interni;
- l'esecuzione di Grid Search e Random Search (inner loop);
- la valutazione del modello selezionato sull'outer test set;
- la misurazione dei tempi di training e inferenza;
- la gestione dei modelli ensemble;
- il salvataggio e il caricamento dei risultati;
- la raccolta delle feature selezionate dalla pipeline.

La metrica principale utilizzata per la selezione del modello è la recall, ma vengono salvate anche precision, accuracy e F1-score.

### `jupyters/eda.ipynb`

Notebook dedicato alla fase di Exploratory Data Analysis. Viene utilizzato per analizzare la struttura del dataset, studiare la distribuzione delle variabili, individuare eventuali pattern, analizzare la presenza di anomalie e valori atipici, osservare il bilanciamento delle classi ed esplorare le relazioni tra le feature e il target.

### `jupyters/pipeline.ipynb`

Notebook dedicato all'esecuzione della Nested Cross-Validation per ciascun modello definito in `config.py`. Prepara i dati, costruisce la pipeline di preprocessing, esegue la Nested Cross-Validation e salva i risultati nel file `data/nested_cv_results.pkl`.

### `jupyters/results_analysis.ipynb`

Notebook dedicato all'analisi dei risultati prodotti da `pipeline.ipynb`. Viene utilizzato per confrontare i modelli e le strategie di ricerca, analizzare le metriche ottenute nei diversi outer fold, studiare i tempi di addestramento e inferenza, analizzare le feature selezionate e le spiegazioni ottenute tramite tecniche di explainability (SHAP).

### `data/dataset.csv`

Dataset utilizzato per il progetto, recuperato tramite `scikit-learn`.

### `data/nested_cv_results.pkl`

File contenente i risultati della Nested Cross-Validation eseguita dal notebook `pipeline.ipynb`.

### `data/shap_results.pkl`

File contenente le spiegazioni SHAP calcolate per ciascuna combinazione di modello e strategia di ricerca.

## Configurazione iniziale

Il progetto utilizza `uv` come package manager.

La configurazione dell'ambiente e delle dipendenze e definita nei file:

- `pyproject.toml`: contiene i metadati del progetto, la versione minima di Python e le dipendenze;
- `uv.lock`: contiene le versioni risolte delle dipendenze, garantendo maggiore riproducibilita dell'ambiente.

Per installare le dipendenze è sufficiente eseguire, dalla root del progetto, il seguente comando:

```bash
uv sync
```

## Ordine di esecuzione

I notebook devono essere eseguiti nel seguente ordine:

1. `jupyters/eda.ipynb`
2. `jupyters/pipeline.ipynb`
3. `jupyters/results_analysis.ipynb`

## Glossario dei risultati

Il file `nested_cv_results.pkl` contiene un `pandas.DataFrame`.

Per ogni modello, per ogni outer fold e per ogni strategia di ricerca (`grid` o `random`), viene salvata una riga contenente:

| Colonna | Descrizione |
|---|---|
| `model` | Nome del modello valutato |
| `best_estimator` | Miglior stimatore ottenuto nell'inner loop |
| `outer_fold` | Indice dell'outer fold |
| `y_true` | Valori reali del target nell'outer test set |
| `y_pred` | Predizioni effettuate sull'outer test set |
| `y_proba` | Probabilita predette per la classe positiva |
| `search_type` | Strategia utilizzata: `grid` oppure `random` |
| `recall` | Recall calcolato sull'outer test set |
| `precision` | Precision calcolata sull'outer test set |
| `accuracy` | Accuracy calcolata sull'outer test set |
| `f1` | F1-score calcolato sull'outer test set |
| `inner_best_score` | Miglior punteggio ottenuto nell'inner loop |
| `inner_mean_validation_score` | Media del punteggio di validazione della configurazione migliore |
| `inner_std_validation_score` | Deviazione standard del punteggio di validazione |
| `fit_time_seconds` | Tempo di addestramento |
| `inference_time_seconds` | Tempo di calcolo delle predizioni |
| `inference_proba_time_seconds` | Tempo di calcolo delle probabilita |
| `best_params` | Migliori iperparametri individuati |
| `train_indexes` | Indici utilizzati per l'outer training set |
| `test_indexes` | Indici utilizzati per l'outer test set |
| `*_selected_features` | Feature selezionate dagli step di feature selection |


Con `n_outer = 5` e due strategie di ricerca, ogni modello produce normalmente:

```text
5 outer fold x 2 strategie di ricerca = 10 righe
```



## Autori
* Raffaele Coppola - [raffaele-coppola](https://github.com/raffaele-coppola/)
* Michele Antonio Annunziata - [micheleantonioannunziata](https://github.com/micheleantonioannunziata/)