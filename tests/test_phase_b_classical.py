import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from src.aggregate_results import INPUT_IDENTITY, check_provenance
from src.evaluate import positive_scores
from src.features import AUDITED_PREDICTORS
from src.models_classical import FeatureSchema, VariancePCA, classical_from_recipe
from src.phase_b_classical import build_fits, feature_schema, make_recipe, paired_deltas, tune_tree
from src.splits import OBSERVED, TIME, get_folds


class PhaseBClassicalTests(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2025-01-01", "2025-10-30", tz="UTC")
        self.meta = pd.DataFrame({TIME: times, OBSERVED: times, "LATE": np.arange(len(times)) % 2},
                                 index=pd.Index([f"id-{i:04}" for i in range(len(times))], name="flight_id"))

    def test_variance_pca_retains_smallest_prefix_of_training_spectrum(self):
        X = csr_matrix(np.random.default_rng(42).normal(size=(80, 12)))
        pca = VariancePCA(variance=.9).fit(X)
        full = PCA(svd_solver="covariance_eigh").fit(X)
        k = pca.n_components_
        self.assertGreaterEqual(pca.explained_variance_ratio_[:k].sum(), .9)
        self.assertLess(pca.explained_variance_ratio_[:k-1].sum(), .9)
        np.testing.assert_allclose(pca.transform(X), full.transform(X)[:, :k], atol=1e-12)

    def test_fit_modes_share_validation_and_keep_exact_training_ids(self):
        plans = build_fits(self.meta, ("sampled", "full_fold"))
        folds = get_folds(self.meta)
        for (sample, left, sampled), (train, right, full) in zip(*plans.values(), strict=True):
            np.testing.assert_array_equal(train, folds[full["fold"] - 1][0])
            np.testing.assert_array_equal(left, right)
            self.assertTrue(set(sample) < set(train))
            self.assertEqual(sampled["validation_ids_sha256"], full["validation_ids_sha256"])

    def test_recipe_round_trip_and_batched_probabilities(self):
        schema = FeatureSchema((("distance", "numeric"), ("origin", "categorical")))
        X = pd.DataFrame({"distance": np.arange(len(self.meta)), "origin": "AAA"}, index=self.meta.index)
        recipe = make_recipe("random_forest", variance=.95, scaled=True, n_estimators=3, max_depth=2)
        train, valid = get_folds(self.meta)[0]
        models = [classical_from_recipe(schema, 42, spec).fit(X.iloc[train], self.meta.LATE.iloc[train])
                  for spec in (recipe, json.loads(json.dumps(recipe)))]
        np.testing.assert_array_equal(positive_scores(models[0], X.iloc[valid], batch_size=7),
                                      positive_scores(models[1], X.iloc[valid]))

    def test_manifest_roles_and_weather_approval(self):
        core = tuple((item["name"], item["role"]) for item in AUDITED_PREDICTORS)
        with self.assertRaisesRegex(ValueError, "audited G1"):
            feature_schema(SimpleNamespace(schema=FeatureSchema(core[:-1])), "G1", False, None)
        weather = SimpleNamespace(schema=FeatureSchema((*core, ("weather_missing", "indicator"))))
        with tempfile.TemporaryDirectory() as temp:
            registry = Path(temp) / "registry.csv"
            registry.write_text("name,group,approval_status,allowed_6h,allowed_1h\nweather_missing,G2,pending,yes,yes\n")
            with self.assertRaisesRegex(ValueError, "not approved"):
                feature_schema(weather, "G2", False, registry)
            registry.write_text(registry.read_text().replace("pending", "approved"))
            self.assertEqual(feature_schema(weather, "G2", False, registry), weather.schema)

    def test_pairing_rejects_mismatched_validation_ids(self):
        keys = dict(checkpoint_minutes=60, fold=1, seed=42, fit_mode="full_fold",
                    train_ids_sha256="train", pr_auc=.3, roc_auc=.6)
        rows = [dict(keys, model="pca", validation_ids_sha256="left"),
                dict(keys, model="control", validation_ids_sha256="right")]
        with self.assertRaisesRegex(ValueError, "mismatched fit IDs"):
            paired_deltas(rows, [("pca", "control")])

    def test_supplemental_pruning_grid_uses_only_the_first_training_draw(self):
        fits = build_fits(self.meta, ("sampled",))
        added = []

        def path(_, train, seed):
            value = .0002 if train is fits["sampled"][0][0] else .0009
            return SimpleNamespace(ccp_alphas=np.array([0, value, 0.01]), impurities=np.zeros(3))

        study = SimpleNamespace(minutes=360, fits=fits, pruning_path=path, paths=[], rank=lambda names: names,
                                score_candidate=lambda name, recipe: added.append(recipe))
        tune_tree(study)
        alphas = {recipe["parameters"].get("classifier__ccp_alpha") for recipe in added}
        self.assertIn(.0002, alphas)
        self.assertNotIn(.0009, alphas)

    def test_aggregation_rejects_same_size_but_different_validation_cohorts(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / name for name in ("p4", "p5")]
            identity = dict.fromkeys(INPUT_IDENTITY, "same")
            for path in paths:
                path.mkdir()
                (path / "experiment.json").write_text(json.dumps(identity))
            keys = dict(checkpoint_minutes=60, fold=1, seed=42, train_rows=9, validation_rows=5,
                        revision="same", fit_mode="same", feature_group="same", train_ids_sha256="train")
            rows = pd.DataFrame([dict(keys, model="rf", validation_ids_sha256="left"),
                                 dict(keys, model="booster", validation_ids_sha256="right")])
            with self.assertRaisesRegex(ValueError, "different train or validation cohorts"):
                check_provenance(*paths, rows)
