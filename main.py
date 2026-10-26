"""Download UCI Wine Quality data and compare transparent classification baselines."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

WINE_URLS = {
    "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}


def quality_class(quality: float, threshold: int = 7) -> int:
    """Map the ordinal quality score to a high-quality indicator."""
    if not 0 <= float(quality) <= 10:
        raise ValueError("quality must be between 0 and 10")
    if not 1 <= int(threshold) <= 10:
        raise ValueError("threshold must be between 1 and 10")
    return int(float(quality) >= int(threshold))


def main(output_dir: str = "outputs", threshold: int = 7, seed: int = 42) -> dict:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    import joblib

    frames = []
    for colour, url in WINE_URLS.items():
        frame = pd.read_csv(url, sep=";")
        frame["wine_colour"] = colour
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    y = data["quality"].map(lambda value: quality_class(value, threshold))
    X = data.drop(columns=["quality"])
    numeric = [name for name in X.columns if name != "wine_colour"]
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("colour", OneHotEncoder(handle_unknown="ignore"), ["wine_colour"]),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                  class_weight="balanced", random_state=seed,
                                                  n_jobs=-1),
    }
    results = {}
    best_name, best_score, best_model = None, -1.0, None
    for name, estimator in models.items():
        model = make_pipeline(preprocess, estimator)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        score = float(balanced_accuracy_score(y_test, pred))
        results[name] = {
            "balanced_accuracy": score,
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
        }
        if score > best_score:
            best_name, best_score, best_model = name, score, model
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({
        "dataset": "UCI Wine Quality", "threshold": threshold,
        "positive_label": f"quality >= {threshold}", "seed": seed,
        "test_rows": int(len(y_test)), "models": results,
        "selected_by_balanced_accuracy": best_name,
    }, indent=2), encoding="utf-8")
    joblib.dump(best_model, out / "best_model.joblib")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--threshold", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(main(args.output_dir, args.threshold, args.seed), indent=2))
