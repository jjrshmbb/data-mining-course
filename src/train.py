"""模型训练: 统一的模型字典 + 交叉验证 + 拟合预测。

想加模型只需在 get_models 里加一行, 主流程与评估代码无需改动。
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target

from . import config


def get_models(task: str) -> dict:
    """返回 {模型名: 未训练的模型对象}。"""
    seed = config.RANDOM_SEED
    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
            "decision_tree": DecisionTreeClassifier(random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "svm": SVC(kernel="rbf", random_state=seed),
        }
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=seed),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
    }


def train_models(X_train, y_train, X_test, y_test, only: str | None = None) -> dict:
    """训练并评估多个模型, 返回 {模型名: 结果字典}。"""
    task = "regression" if type_of_target(y_train) in ("continuous", "continuous-multioutput") else "classification"
    models = get_models(task)
    if only:
        if only not in models:
            raise KeyError(f"未知模型 '{only}', 可选: {list(models)}")
        models = {only: models[only]}

    cv = (StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
          if task == "classification"
          else KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED))
    scoring = "accuracy" if task == "classification" else "r2"

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_score = None
        if hasattr(model, "predict_proba"):
            try:
                y_score = model.predict_proba(X_test)
            except Exception:
                y_score = None
        if y_score is None and hasattr(model, "decision_function"):
            try:
                y_score = model.decision_function(X_test)
            except Exception:
                y_score = None

        results[name] = {
            "model": model,
            "task": task,
            "y_pred": y_pred,
            "y_score": y_score,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
        }
        print(f"      {name:20s} 交叉验证 {scoring} = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results
