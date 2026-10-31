"""Train and compare Decision Tree, Random Forest and XGBoost on the heart dataset.

    python train.py

Writes the best model to models/best_model.joblib and all results to
models/results.json (both are read by the Streamlit app).
"""
import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import features as F

SEED = 42

CANDIDATES = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"model__max_depth": [2, 3, 4, 5, 8], "model__min_samples_split": [2, 10, 30, 50]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [4, 8, None],
            "model__min_samples_split": [2, 10, 30],
        },
    ),
    "XGBoost": (
        XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0),
        {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.1],
        },
    ),
}


def main():
    df = F.load_data()
    X, y = df[F.FEATURES], df[F.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, (estimator, grid) in CANDIDATES.items():
        search = GridSearchCV(F.make_pipeline(estimator), grid, cv=cv,
                              scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred, proba = best.predict(X_test), best.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_accuracy": round(search.best_score_, 4),
            "train_accuracy": round(accuracy_score(y_train, best.predict(X_train)), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_recall": round(recall_score(y_test, pred), 4),
            "test_f1": round(f1_score(y_test, pred), 4),
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        }
        fitted[name] = best
        print(f"{name:14s} CV {results[name]['cv_accuracy']:.3f} | test {results[name]['test_accuracy']:.3f} "
              f"| AUC {results[name]['test_roc_auc']:.3f} | {results[name]['best_params']}")

    # Choose the winner by cross-validation score (NOT by test score, to keep the test set unbiased)
    winner = max(results, key=lambda n: results[n]["cv_accuracy"])
    best_pipe = fitted[winner]

    # Feature importances (works for all three tree-based models)
    names = [n.split("__", 1)[1] for n in best_pipe.named_steps["prep"].get_feature_names_out()]
    importances = best_pipe.named_steps["model"].feature_importances_
    top = sorted(zip(names, importances), key=lambda t: -t[1])[:10]

    out = {
        "winner": winner,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "models": results,
        "top_features": [{"feature": n, "importance": round(float(v), 4)} for n, v in top],
    }
    joblib.dump(best_pipe, F.MODEL_PATH)
    F.RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nBest model (by CV): {winner}. Saved to {F.MODEL_PATH.name} and {F.RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
