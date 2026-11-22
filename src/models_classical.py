"""Phase A classical model factories and fold-local preprocessing."""

import re
from dataclasses import dataclass
from functools import partial
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.sparse.linalg import ArpackError
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.sparsefuncs import min_max_axis

PCA_COMPONENTS = 2
MISSING_CATEGORY = "__P4_MISSING__"
ROLES = ("numeric", "categorical", "indicator")

# Reserved metadata and raw target-flight outcome fields from the BTS audit.
_RESERVED = frozenset("""
    flightid late label target outcome scheduleddepartureutc actualdepartureutc
    scheduledarrivalutc actualarrivalutc deptime depdelay depdelayminutes depdel15
    departuredelay departuredelaygroups taxiout wheelsoff wheelson
    taxiin arrtime arrdelay arrdelayminutes arrdel15 arrivaldelay arrivaldelaygroups
    cancelled cancellationcode diverted actualelapsedtime airtime carrierdelay
    weatherdelay nasdelay securitydelay lateaircraftdelay firstdeptime totaladdgtime
    longestaddgtime divairportlandings divreacheddest divactualelapsedtime divarrdelay
    divdistance actualdeptime actualarrtime actualdeparturetime actualarrivaltime
""".split())
_DIVERSION_OUTCOME = re.compile(r"div[1-5](?:airport(?:id|seqid)?|wheelson|wheelsoff|"
                                r"totalgtime|longestgtime|tailnum)")


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered (name, role) pairs from an approved feature specification."""

    predictors: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not self.predictors:
            raise ValueError("Declare at least one predictor")
        for name, role in self.predictors:
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError(f"Invalid predictor name {name!r}")
            key = re.sub(r"[^a-z0-9]", "", name.casefold())
            unprefixed = key.removeprefix("targetflight").removeprefix("target")
            if key in _RESERVED or unprefixed in _RESERVED or _DIVERSION_OUTCOME.fullmatch(unprefixed):
                raise ValueError(f"Reserved metadata or outcome column {name!r}")
            if role not in ROLES:
                raise ValueError(f"Unknown role {role!r} for {name!r}; use {ROLES}")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Duplicate predictor names")

    @property
    def columns(self):
        return [name for name, _ in self.predictors]

    def by_role(self, role):
        return [name for name, declared in self.predictors if declared == role]

    def validate_columns(self, columns):
        actual = pd.Index(columns)
        if actual.has_duplicates:
            raise ValueError("Duplicate predictor columns in input")
        missing = pd.Index(self.columns).difference(actual, sort=False).tolist()
        extra = actual.difference(self.columns, sort=False).tolist()
        if missing or extra:
            raise ValueError(f"Predictor schema mismatch; missing={missing}, unexpected={extra}")

    def normalize(self, frame):
        """Coerce individual values without learning any training statistics."""
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Classical predictors must be a pandas DataFrame")
        self.validate_columns(frame.columns)
        columns = {}
        for name, role in self.predictors:
            values = frame[name]
            if role == "categorical":
                kind = pd.api.types.infer_dtype(values, skipna=True)
                if kind not in {"string", "integer", "floating", "mixed-integer-float",
                                "boolean", "empty"}:
                    # Heterogeneous scalar categories are also normalized to strings.
                    if not values.dropna().map(
                        lambda value: isinstance(value, (str, int, float, np.number, np.bool_))
                        and not np.iscomplexobj(value)
                    ).all():
                        raise ValueError(f"{name!r} must contain scalar string or numeric categories")
                values = values.astype("string").astype(object)
                if values.eq(MISSING_CATEGORY).any():
                    raise ValueError(f"{name!r} contains reserved missing token {MISSING_CATEGORY!r}")
                columns[name] = values.where(values.notna(), np.nan)
                continue

            try:
                numbers = pd.to_numeric(values, errors="raise")
                if (pd.api.types.is_complex_dtype(numbers)
                        or pd.api.types.infer_dtype(values, skipna=True)
                        in {"datetime", "datetime64", "date", "timedelta", "timedelta64"}):
                    raise ValueError("Expected real numbers")
                numbers = numbers.to_numpy(dtype=np.float64, na_value=np.nan)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name!r} must contain real numeric values or nulls") from exc
            if (np.isnan(numbers) & values.notna().to_numpy()).any():
                raise ValueError(f"{name!r} must contain real numeric values or nulls")
            if np.isinf(numbers).any():
                raise ValueError(f"{name!r} contains infinite values")
            if role == "indicator" and not np.isin(numbers, [0, 1]).all():
                raise ValueError(f"{name!r} must contain nonmissing binary indicators (0 or 1)")
            columns[name] = numbers
        return pd.DataFrame(columns, index=frame.index)


class CheckedPCA(PCA):
    """Centered ARPACK PCA with checks on the actual fold's training matrix."""

    def _checked_fit(self, fit, X, y):
        k = self.n_components
        rows, width = X.shape
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 < k < min(rows, width):
            raise ValueError(
                f"PCA n_components={k} requires 0 < k < min(training rows, encoded width); "
                f"received {rows} rows and {width} columns. Use a smaller k or supply more "
                "training rows or usable predictors."
            )
        if issparse(X):
            minimum, maximum = min_max_axis(X, axis=0)
        else:
            minimum, maximum = X.min(axis=0), X.max(axis=0)
        if not np.any(maximum > minimum):
            raise ValueError(f"PCA n_components={k} received constant training input. "
                             "Supply predictors with variation in this fold.")
        try:
            result = fit(X, y)
        except (ArpackError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"ARPACK PCA failed for n_components={k}. "
                             "Check the training matrix or try a smaller k.") from exc
        if not (np.isfinite(self.explained_variance_).all()
                and np.isfinite(self.explained_variance_ratio_).all()):
            raise ValueError(f"PCA n_components={k} returned nonfinite explained variance. "
                             "Check the scale and variation of the training predictors.")
        self.training_shape_ = (rows, width)
        nonzero = X.count_nonzero() if issparse(X) else np.count_nonzero(X)
        self.training_density_ = nonzero / (rows * width)
        return result

    def fit(self, X, y=None):
        return self._checked_fit(super().fit, X, y)

    def fit_transform(self, X, y=None):
        return self._checked_fit(super().fit_transform, X, y)


def _preprocessor(schema, *, scale_numeric=False):
    numeric = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric), schema.by_role("numeric")),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY,
                                         keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         max_categories=128, sparse_output=True)),
            ]), schema.by_role("categorical")),
            ("indicator", "passthrough", schema.by_role("indicator")),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def _pipeline(schema, classifier, *, scale_numeric=False, pca=None):
    steps = [
        ("normalize", FunctionTransformer(schema.normalize)),
        ("preprocess", _preprocessor(schema, scale_numeric=scale_numeric)),
    ]
    if pca is not None:
        steps.append(("pca", pca))
    return Pipeline([*steps, ("classifier", classifier)])


def majority(seed):
    return DummyClassifier(strategy="most_frequent", random_state=seed)


def decision_tree(seed, *, schema):
    return _pipeline(schema, DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=25, ccp_alpha=0, random_state=seed,
    ))


def random_forest(seed, *, schema):
    return _pipeline(schema, RandomForestClassifier(
        n_estimators=32, max_depth=12, min_samples_leaf=10, max_features="sqrt",
        class_weight=None, n_jobs=1, random_state=seed,
    ))


def _logistic(seed, schema, pca):
    return _pipeline(schema, LogisticRegression(
        solver="newton-cg", C=1, max_iter=1000, class_weight=None, random_state=seed,
    ), scale_numeric=True, pca=pca)


def logistic(seed, *, schema):
    return _logistic(seed, schema, "passthrough")


def pca_logistic(seed, *, schema):
    return _logistic(seed, schema, CheckedPCA(
        n_components=PCA_COMPONENTS, svd_solver="arpack", whiten=False, random_state=seed,
    ))


def classical_factories(schema):
    """Bind the input schema to factories taking only the evaluator's seed."""
    return {
        "majority": majority,
        "decision_tree": partial(decision_tree, schema=schema),
        "random_forest": partial(random_forest, schema=schema),
        "logistic": partial(logistic, schema=schema),
        "pca_logistic": partial(pca_logistic, schema=schema),
    }
