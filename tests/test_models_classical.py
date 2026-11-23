import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import ArpackError
from sklearn.decomposition import PCA

from src.evaluate import SEEDS, evaluate
from src.models_classical import MISSING_CATEGORY, FeatureSchema, classical_factories
from src.run_classical import (
    METRICS, checkpoint_features, evaluate_configuration, fold_details, main, paired_metadata,
    read_manifest, schedule_sample,
)
from src.splits import OBSERVED, TIME, get_folds

SCHEMA = FeatureSchema((
    ("distance", "numeric"), ("empty", "numeric"), ("carrier", "categorical"),
    ("available", "indicator"),
))


def cohort():
    times = pd.to_datetime(
        ["2025-01-15", "2025-06-25", "2025-07-15", "2025-08-15", "2025-09-15", "2025-10-15"],
        utc=True,
    ).repeat(8)
    index = pd.Index([f"flight-{i:02d}" for i in range(len(times))], name="flight_id")
    values = np.arange(len(index))
    labels = values % 2
    development = pd.DataFrame({
        TIME: times, OBSERVED: times + pd.to_timedelta(15 * labels, unit="min"), "LATE": labels,
    }, index=index)
    X = pd.DataFrame({
        "distance": pd.array(np.where(values % 4, values, np.nan), dtype="Float64"),
        "empty": pd.array(np.where(values < 16, np.nan, values), dtype="Float64"),
        "carrier": pd.array(
            np.where(values < 16, np.where(values % 3, "AA", "UA"), "NEW"), dtype="string[pyarrow]",
        ),
        "available": pd.array((values % 3) != 0, dtype="boolean"),
    }, index=index)
    X.loc[index[16], "carrier"] = pd.NA
    return X, development


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.X, self.development = cohort()
        self.factories = classical_factories(SCHEMA)
        self.train, self.valid = get_folds(self.development)[0]

    def test_all_factories_produce_probabilities_in_class_order(self):
        X_train, X_valid = self.X.iloc[self.train], self.X.iloc[self.valid]
        y_train = self.development["LATE"].iloc[self.train]
        for name, factory in self.factories.items():
            with self.subTest(model=name):
                estimator = factory(SEEDS[0])
                self.assertIsNot(estimator, factory(SEEDS[0]))
                self.assertFalse(hasattr(estimator, "classes_"))
                estimator.fit(X_train, y_train)
                np.testing.assert_array_equal(estimator.classes_, [0, 1])
                proba = estimator.predict_proba(X_valid)
                self.assertEqual(proba.shape, (len(X_valid), 2))
                self.assertTrue(np.isfinite(proba).all())
                self.assertTrue(((0 <= proba) & (proba <= 1)).all())
                np.testing.assert_allclose(proba.sum(axis=1), 1)

    def test_logistic_preprocessing_is_matched_and_train_only(self):
        logistic = self.factories["logistic"](42)
        pca_logistic = self.factories["pca_logistic"](42)
        self.assertEqual(
            {key: repr(value) for key, value in logistic["preprocess"].get_params().items()},
            {key: repr(value) for key, value in pca_logistic["preprocess"].get_params().items()},
        )
        self.assertEqual(logistic["classifier"].get_params(), pca_logistic["classifier"].get_params())
        self.assertEqual(logistic["pca"], "passthrough")
        fitted = []

        def evaluate_first_fold(X):
            def capture(seed):
                model = self.factories["pca_logistic"](seed)
                if not fitted:
                    fitted.append(model)
                return model

            evaluate(capture, X, self.development, seeds=(42,))
            model = fitted.pop()
            prep = model["preprocess"]
            numeric = prep.named_transformers_["numeric"]
            encoder = prep.named_transformers_["categorical"]["encode"]
            self.assertTrue(issparse(prep.transform(model["normalize"].transform(X.iloc[self.train]))))
            np.testing.assert_array_equal(encoder.categories_[0], ["AA", "UA"])
            self.assertEqual(numeric["impute"].statistics_[1], 0)
            return (numeric["impute"].statistics_, numeric["scale"].mean_, numeric["scale"].scale_,
                    model["pca"].mean_, model["pca"].explained_variance_)

        changed = self.X.copy()
        changed.loc[self.X.index[self.valid], ["distance", "empty", "carrier"]] = [
            10_000, -2_000, "VALIDATION_ONLY",
        ]
        for original, mutated in zip(evaluate_first_fold(self.X), evaluate_first_fold(changed)):
            np.testing.assert_allclose(original, mutated)

    def test_normalization_rejects_malformed_inputs(self):
        self.assertTrue(np.isnan(SCHEMA.normalize(self.X.iloc[:1]).loc["flight-00", "distance"]))
        for column, value, message in [
            ("available", pd.NA, "nonmissing binary"),
            ("available", 2, "nonmissing binary"),
            ("distance", np.inf, "infinite"),
            ("distance", "broken", "real numeric"),
            ("distance", "", "real numeric"),
            ("carrier", MISSING_CATEGORY, "reserved missing token"),
            ("carrier", ["nested"], "scalar"),
        ]:
            with self.subTest(column=column, value=value):
                bad = self.X.astype(object)
                bad.at[bad.index[0], column] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.factories["decision_tree"](42).fit(bad, self.development["LATE"])

    def test_pca_rejects_illegal_shape_constant_data_and_solver_failure(self):
        pca = self.factories["pca_logistic"](42)["pca"]
        for data in (np.ones((2, 4)), csr_matrix(np.ones((8, 2)))):
            with self.assertRaisesRegex(ValueError, "n_components=2.*training rows, encoded width"):
                pca.fit_transform(data)
        with self.assertRaisesRegex(ValueError, "constant training input"):
            pca.fit(csr_matrix(np.full((6, 4), 0.1)))
        with patch.object(PCA, "fit_transform", side_effect=ArpackError(-1)):
            with self.assertRaisesRegex(ValueError, "ARPACK PCA failed.*n_components=2"):
                pca.fit_transform(csr_matrix(np.eye(4)))

    def test_pca_runner_error_names_checkpoint_fold_and_seed(self):
        X = self.X.assign(distance=0.0, empty=0.0, available=False, carrier="AA")
        with self.assertRaisesRegex(ValueError, r"pca_logistic at 360 min, fold 1, seed 42.*constant"):
            evaluate_configuration("pca_logistic", self.factories["pca_logistic"], X,
                                   self.development, 360, fold_details(self.development))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.X, self.development = cohort()
        self.document = {
            "development": {"path": str(self.directory / "dev.parquet"), "release": "synthetic"},
            "feature_spec_revision": "test-revision",
            "predictors": [{"name": name, "role": role} for name, role in SCHEMA.predictors],
            "checkpoints": [
                {"path": str(self.directory / f"{offset}.parquet"), "offset_minutes": offset,
                 "source_release": "synthetic"}
                for offset in (360, 60)
            ],
        }
        self.manifest = self.directory / "inputs.json"
        self.development.sample(frac=1, random_state=5).to_parquet(self.document["development"]["path"])
        for checkpoint in self.document["checkpoints"]:
            self.X.sample(frac=1, random_state=checkpoint["offset_minutes"]).to_parquet(checkpoint["path"])

    def inputs(self):
        self.manifest.write_text(json.dumps(self.document))
        return read_manifest(self.manifest)

    def test_shuffled_flights_are_aligned_and_paired_run_has_exact_metric_keys(self):
        inputs = self.inputs()
        development = paired_metadata(inputs)
        pd.testing.assert_frame_equal(development, self.development)
        for checkpoint in inputs.checkpoints:
            X = checkpoint_features(checkpoint, inputs.schema, development.index)
            pd.testing.assert_frame_equal(X, SCHEMA.normalize(self.X))
        output = self.directory / "output"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--input-manifest", str(self.manifest), "--output", str(output)])
        runs = pd.read_csv(output / "runs.csv")
        summary = pd.read_csv(output / "summary.csv")
        expected_keys = {
            (model, offset, fold, seed)
            for model in classical_factories(SCHEMA) for offset in (360, 60)
            for fold in range(1, 5) for seed in SEEDS
        }
        keys = ["model", "checkpoint_minutes", "fold", "seed"]
        self.assertEqual(len(runs), len(expected_keys))
        self.assertEqual(set(runs[keys].itertuples(index=False, name=None)), expected_keys)
        self.assertTrue(np.isfinite(runs[METRICS]).all().all())
        self.assertEqual(len(summary), 10)
        for row in summary.itertuples(index=False):
            scores = runs.loc[(runs.model == row.model)
                              & (runs.checkpoint_minutes == row.checkpoint_minutes), METRICS]
            for metric in METRICS:
                self.assertAlmostEqual(getattr(row, f"{metric}_mean"), scores[metric].mean())
                self.assertAlmostEqual(getattr(row, f"{metric}_std"), scores[metric].std(ddof=0))
        majority = runs[runs.model == "majority"]
        np.testing.assert_array_equal(majority.roc_auc, 0.5)
        np.testing.assert_array_equal(majority.pr_auc, 0.5)
        pca = pd.read_csv(output / "pca_diagnostics.csv")
        self.assertEqual(len(pca), 48)
        self.assertTrue(np.isfinite(pca.cumulative_variance_ratio).all())
        self.assertTrue(pca.encoded_density.between(0, 1, inclusive="neither").all())

    def test_bad_checkpoint_ids_are_rejected_before_fit(self):
        inputs = self.inputs()
        path = inputs.checkpoints[1].path
        for index, message in [
            (self.X.index[:-1], "different flight IDs"),
            (self.X.index[:-1].append(self.X.index[:1]), "duplicate flight IDs"),
            (self.X.index[:-1].append(pd.Index([None], name="flight_id")), "nonmissing string flight IDs"),
        ]:
            with self.subTest(message=message):
                self.X.iloc[:len(index)].set_axis(index).to_parquet(path)
                with self.assertRaisesRegex(ValueError, message):
                    paired_metadata(inputs)
        unknown = self.X.rename(index={self.X.index[-1]: "not-in-p1"})
        for checkpoint in inputs.checkpoints:
            unknown.to_parquet(checkpoint.path)
        with self.assertRaisesRegex(ValueError, "absent from P1 development"):
            paired_metadata(inputs)

    def test_declared_and_undeclared_outcome_columns_are_rejected(self):
        for column in ("LATE", TIME, OBSERVED, "DepDelay", "Target_Flight_ArrDelay", "Div2WheelsOn"):
            with self.subTest(column=column):
                self.document["predictors"].append({"name": column, "role": "numeric"})
                with self.assertRaisesRegex(ValueError, "Reserved metadata or outcome column"):
                    self.inputs()
                self.document["predictors"].pop()
        inputs = self.inputs()
        self.X.assign(DepDelay=15).to_parquet(inputs.checkpoints[0].path)
        with self.assertRaisesRegex(ValueError, "unexpected=.*DepDelay"):
            paired_metadata(inputs)

    def test_incompatible_offsets_releases_and_parquet_declarations(self):
        entry = self.document["checkpoints"][1]
        for offset, message in [(0, "positive integer"), (True, "positive integer"),
                                (360, "distinct checkpoint"), (1441, "exceeds P1")]:
            entry["offset_minutes"] = offset
            with self.assertRaisesRegex(ValueError, message):
                self.inputs()
        entry["offset_minutes"] = 60
        entry["source_release"] = "other"
        with self.assertRaisesRegex(ValueError, "source release differs"):
            self.inputs()
        entry["source_release"] = "synthetic"
        inputs = self.inputs()
        table = pa.Table.from_pandas(self.X)
        pq.write_table(table.replace_schema_metadata({b"checkpoint_minutes": b"45"}),
                       inputs.checkpoints[1].path)
        with self.assertRaisesRegex(ValueError, "checkpoint_minutes does not match"):
            paired_metadata(inputs)

    def test_sampled_id_list_and_frozen_development_dates_are_checked(self):
        sample = self.directory / "sampled.parquet"
        self.document["sampled_ids"] = str(sample)
        self.X.index[:-1].to_frame(index=False).to_parquet(sample, index=False)
        inputs = self.inputs()
        with self.assertRaisesRegex(ValueError, "sampled_ids.*different flight IDs"):
            paired_metadata(inputs)
        self.X.index.to_frame(index=False).to_parquet(sample, index=False)
        invalid = self.development.copy()
        invalid.loc[invalid.index[0], TIME] = pd.Timestamp("2025-11-02", tz="UTC")
        invalid.to_parquet(inputs.development)
        with self.assertRaisesRegex(ValueError, "frozen development period"):
            paired_metadata(inputs)

    def test_schedule_sample_selection_ignores_file_order_and_labels(self):
        schedule = self.development.assign(
            CRSDepTime=2400, CRSElapsedTime=120, Distance=500,
            Reporting_Airline="AA", Origin="JFK", Dest="ORD",
        )
        path = self.directory / "schedule.parquet"
        with patch("src.run_classical.SAMPLE_PER_MONTH", 2):
            schedule.to_parquet(path)
            original_X, original_dev = schedule_sample(path)
            schedule.assign(LATE=1 - schedule["LATE"]).sample(frac=1, random_state=22).to_parquet(path)
            shuffled_X, shuffled_dev = schedule_sample(path)
        pd.testing.assert_frame_equal(original_X, shuffled_X)
        self.assertTrue(original_X.index.equals(original_dev.index))
        self.assertTrue(shuffled_X.index.equals(shuffled_dev.index))
        self.assertEqual(original_dev.groupby(original_dev[TIME].dt.month).size().tolist(),
                         [2, 2, 2, 2, 2, 2])
        np.testing.assert_array_equal(original_X["scheduled_local_minutes"], 0)
        np.testing.assert_array_equal(original_dev["LATE"], 1 - shuffled_dev["LATE"])


if __name__ == "__main__":
    unittest.main()
