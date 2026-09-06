from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from scipy.stats import randint, uniform, loguniform
from imblearn.pipeline import Pipeline
import steps



ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATASET_PATH = DATA_DIR / "dataset.csv"
TEST_SET_PATH = DATA_DIR / "test_set.csv"
NESTED_RESULTS_FILENAME = "nested_cv_results.pkl"

for value in list(globals().values()):
    if isinstance(value, Path) and not value.exists() and not value.suffix:
        value.mkdir(parents=True, exist_ok=True)

DIAGNOSIS_PALETTE = {
    "Malignant": "lightcoral",
    "Benign": "lightgreen"
}

SEARCH_TYPE_PALETTE = {
    "grid": "#2C3E50",
    "random": "#C0392B"
}


RANDOM_STATE = 42

def make_ensemble_branch(classifier):
    return Pipeline(steps=[
        ("welch", steps.WelchSelector()),
        ("clf", classifier)
    ])

models = {
    # modelli base
    "SVC": SVC(random_state=RANDOM_STATE),
    "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE),

    # ensemble
    "VotingClassifier": VotingClassifier(
        estimators=[
            ("SVC", make_ensemble_branch(SVC(random_state=RANDOM_STATE))),
            ("RandomForestClassifier", make_ensemble_branch(RandomForestClassifier(random_state=RANDOM_STATE))),
            ("KNeighborsClassifier", make_ensemble_branch(KNeighborsClassifier())),
            ("LogisticRegression", make_ensemble_branch(LogisticRegression(random_state=RANDOM_STATE))),
        ],
        voting="soft",
        n_jobs=-1,
    ),
    "StackingClassifier": StackingClassifier(
        estimators=[
            ("SVC", make_ensemble_branch(SVC(random_state=RANDOM_STATE))),
            ("RandomForestClassifier", make_ensemble_branch(RandomForestClassifier(random_state=RANDOM_STATE))),
            ("KNeighborsClassifier", make_ensemble_branch(KNeighborsClassifier())),
            ("LogisticRegression", make_ensemble_branch(LogisticRegression(random_state=RANDOM_STATE))),
        ],
        final_estimator=Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=RANDOM_STATE)),
            ]
        ),
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
}


param_grids = {
    "SVC": {
        "grid": [
            {
                "model__kernel": ["linear"],
                "model__C": [0.1, 1, 10, 100],
            },
            {
                "model__kernel": ["rbf"],
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
            },
        ],
        "random": [
            {
                "model__kernel": ["linear"],
                "model__C": loguniform(0.01, 100.0),
            },
            {
                "model__kernel": ["rbf"],
                "model__C": loguniform(0.01, 100.0),
                "model__gamma": loguniform(0.001, 1.0),
            },
        ],
    },

    "RandomForestClassifier": {
        "grid": {
            "model__n_estimators": [100, 150],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__min_samples_split": [4, 8],
            "model__min_samples_leaf": [2, 4],
            "model__max_features": [0.5, 0.7],
            "model__max_samples": [0.8, 1.0],
        },
        "random": {
            "model__n_estimators": randint(50, 151),  
            "model__max_depth": [2, 4, 6, 8],  
            "model__min_samples_split": randint(4, 13), 
            "model__min_samples_leaf": randint(2, 7),  
            "model__max_features": uniform(0.4, 0.6),  
            "model__max_samples": uniform(0.7, 0.3),  
        }
    },


    "KNeighborsClassifier": {
        "grid": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
        "random": {
            "model__n_neighbors": randint(3, 21),
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
    },

    "LogisticRegression": {
        "grid": [
            {
                "model__solver": ["lbfgs"],
                "model__l1_ratio": [0],
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__max_iter": [1000],
            },
            {
                "model__solver": ["saga"],
                "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__max_iter": [2000],
            },
        ],
        "random": [
            {
                "model__solver": ["lbfgs"],
                "model__l1_ratio": [0],
                "model__C": loguniform(0.01, 100.0),
                "model__max_iter": [1000],
            },
            {
                "model__solver": ["saga"],
                "model__l1_ratio": uniform(0, 1),
                "model__C": loguniform(0.01, 100.0),
                "model__max_iter": [2000],
            },
        ],
    },

    "StackingClassifier": {
        "grid": [
        {
            "model__final_estimator__classifier__solver": ["lbfgs"],
            "model__final_estimator__classifier__l1_ratio": [0.0],
            "model__final_estimator__classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__final_estimator__classifier__max_iter": [1000],
        },
        {
            "model__final_estimator__classifier__solver": ["saga"],
            "model__final_estimator__classifier__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            "model__final_estimator__classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__final_estimator__classifier__max_iter": [2000],
        },
        ],
        "random": [
            {
                "model__final_estimator__classifier__solver": ["lbfgs"],
                "model__final_estimator__classifier__l1_ratio": [0.0],
                "model__final_estimator__classifier__C": loguniform(0.01, 100.0),
                "model__final_estimator__classifier__max_iter": [1000],
            },
            {
                "model__final_estimator__classifier__solver": ["saga"],
                "model__final_estimator__classifier__l1_ratio": uniform(0.0, 1.0),
                "model__final_estimator__classifier__C": loguniform(0.01, 100.0),
                "model__final_estimator__classifier__max_iter": [2000],
            },
        ],
    }
}