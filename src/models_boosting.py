"""Fixed Phase A smoke configurations. Each call builds a fresh fold-local pipeline."""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.models_classical import _preprocessor

MODELS = ("adaboost", "gradient_boosting", "xgboost", "lightgbm", "catboost")


def make_booster(name, schema, seed):
    constructors = {
        "adaboost": lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1), n_estimators=30, random_state=seed),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=30, max_depth=2, random_state=seed),
        "xgboost": lambda: XGBClassifier(
            n_estimators=30, max_depth=3, tree_method="hist", n_jobs=1, random_state=seed),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=30, num_leaves=15, n_jobs=1, verbosity=-1, random_state=seed),
        "catboost": lambda: CatBoostClassifier(
            iterations=30, depth=4, thread_count=1, verbose=False, allow_writing_files=False,
            random_seed=seed),
    }
    return make_pipeline(FunctionTransformer(schema.normalize), _preprocessor(schema), constructors[name]())
