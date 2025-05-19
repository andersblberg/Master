# ────────────────────────────────────────────────────────────────
# Parallel Random-Forest helper (all CPU cores)
# ────────────────────────────────────────────────────────────────
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_rf_par_model(
    X,
    y,
    *,
    n_estimators: int = 100,
    test_size: float = 0.20,
    random_state: int = 42,
    n_jobs: int = -1,  # <─ uses every core
    **rf_kwargs,
):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        **rf_kwargs,
    )
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return {"model": clf, "accuracy": accuracy}
