# src/models.py

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .config import ART, RESULTS  # assumes these are defined in config.py


def load_splits():
    """
    Load preprocessed design matrices and targets from artifacts/.
    Uses:
      - X_train.npy, X_val.npy, X_test.npy
      - y_train.npy, y_val.npy, y_test.npy
      - feature_columns_after_preprocess.json (or feature_columns.json)
    """
    X_train = np.load(ART / "X_train.npy")
    X_val = np.load(ART / "X_val.npy")
    X_test = np.load(ART / "X_test.npy")

    y_train = np.load(ART / "y_train.npy")
    y_val = np.load(ART / "y_val.npy")
    y_test = np.load(ART / "y_test.npy")

    feat_after = ART / "feature_columns_after_preprocess.json"
    feat_before = ART / "feature_columns.json"

    if feat_after.exists():
        with open(feat_after, "r") as f:
            feature_names = json.load(f)
    elif feat_before.exists():
        with open(feat_before, "r") as f:
            feature_names = json.load(f)
    else:
        raise FileNotFoundError(
            "Could not find feature_columns_after_preprocess.json or feature_columns.json"
        )

    # basic sanity checks
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == len(feature_names)
    assert X_val.shape[1] == len(feature_names)
    assert X_test.shape[1] == len(feature_names)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def make_model(model_name: str):
    """
    Return a fresh sklearn regressor instance for the given name.
    All hyperparameters are simple, reasonable defaults.
    """
    if model_name == "linear":
        return LinearRegression()
    elif model_name == "ridge":
        # you can later make alpha a hyperparameter if you want
        return Ridge(alpha=1.0)
    elif model_name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=0,
        )
    elif model_name == "gbr":
        return GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=0,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name!r}")


def run_cv_and_test():
    """
    Main modeling routine.

    Steps:
      1. Load X/y splits and feature names.
      2. Combine train + val into an 'other' set for cross-validation.
      3. Run RepeatedKFold CV for 4 models:
            - LinearRegression
            - Ridge
            - RandomForestRegressor
            - GradientBoostingRegressor
         Record RMSE and R² for each CV split.
      4. Compute a simple baseline that always predicts the mean of y_other.
      5. Save:
            - cv_scores.csv       (one row per model x CV split)
            - cv_summary.csv      (mean ± std per model)
            - test_predictions.csv (best model on held-out test)
            - baseline_vs_best.txt (baseline vs best model test metrics)
    """
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_splits()

    print("X_train shape:", X_train.shape)
    print("X_val   shape:", X_val.shape)
    print("X_test  shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val   shape:", y_val.shape)
    print("y_test  shape:", y_test.shape)
    print("n_features:", len(feature_names))

    RESULTS.mkdir(parents=True, exist_ok=True)

    # combine train + val for CV and final training
    X_other = np.vstack([X_train, X_val])
    y_other = np.concatenate([y_train, y_val])

    # -------------------------------------------------
    # 1. Baseline: always predict mean of y_other
    # -------------------------------------------------
    baseline_mean = float(np.mean(y_other))
    print(f"\nBaseline: constant prediction = {baseline_mean:.4f} eV")

    baseline_pred_test = np.full_like(y_test, fill_value=baseline_mean, dtype=float)
    baseline_rmse_test = mean_squared_error(y_test, baseline_pred_test, squared=False)
    baseline_r2_test = r2_score(y_test, baseline_pred_test)

    print(f"Baseline Test RMSE: {baseline_rmse_test:.4f}")
    print(f"Baseline Test R²  : {baseline_r2_test:.4f}")

    # -------------------------------------------------
    # 2. RepeatedKFold CV over models
    # -------------------------------------------------
    model_names = ["linear", "ridge", "rf", "gbr"]

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=0,
    )

    rows = []
    split_index = 0

    for model_name in model_names:
        print(f"\n==== Model: {model_name} ====")
        rmse_list = []
        r2_list = []

        split_index = 0

        for train_idx, val_idx in rkf.split(X_other, y_other):
            X_tr = X_other[train_idx]
            y_tr = y_other[train_idx]

            X_va = X_other[val_idx]
            y_va = y_other[val_idx]

            # standardize features based on training fold
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_va_scaled = scaler.transform(X_va)

            model = make_model(model_name)
            model.fit(X_tr_scaled, y_tr)

            y_va_pred = model.predict(X_va_scaled)

            rmse = mean_squared_error(y_va, y_va_pred, squared=False)
            r2 = r2_score(y_va, y_va_pred)

            rmse_list.append(rmse)
            r2_list.append(r2)

            row = {
                "model": model_name,
                "cv_split": split_index,
                "rmse": rmse,
                "r2": r2,
            }
            rows.append(row)

            split_index += 1

        rmse_mean = float(np.mean(rmse_list))
        rmse_std = float(np.std(rmse_list))
        r2_mean = float(np.mean(r2_list))
        r2_std = float(np.std(r2_list))

        print(f"RMSE: mean={rmse_mean:.4f} ± {rmse_std:.4f}")
        print(f"R²  : mean={r2_mean:.4f} ± {r2_std:.4f}")

    # save all CV scores
    cv_df = pd.DataFrame(rows)
    cv_path = RESULTS / "cv_scores.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"\nSaved CV scores to: {cv_path}")

    # summarize per model
    summary_rows = []
    for model_name in model_names:
        df_m = cv_df[cv_df["model"] == model_name]
        rmse_mean = float(df_m["rmse"].mean())
        rmse_std = float(df_m["rmse"].std())
        r2_mean = float(df_m["r2"].mean())
        r2_std = float(df_m["r2"].std())

        summary_rows.append(
            {
                "model": model_name,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "r2_mean": r2_mean,
                "r2_std": r2_std,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved CV summary to: {summary_path}")
    print("\nCV summary:")
    print(summary_df)

    # -------------------------------------------------
    # 3. Pick best model (by lowest mean RMSE) and eval on test
    # -------------------------------------------------
    best_row = summary_df.sort_values("rmse_mean", ascending=True).iloc[0]
    best_model_name = str(best_row["model"])
    best_rmse_mean = float(best_row["rmse_mean"])
    best_rmse_std = float(best_row["rmse_std"])

    print(f"\nBest model by CV RMSE: {best_model_name}")
    print(f"Best CV RMSE: {best_rmse_mean:.4f} ± {best_rmse_std:.4f}")

    # retrain best model on entire X_other, then evaluate on X_test
    scaler = StandardScaler()
    X_other_scaled = scaler.fit_transform(X_other)
    X_test_scaled = scaler.transform(X_test)

    best_model = make_model(best_model_name)
    best_model.fit(X_other_scaled, y_other)

    y_test_pred = best_model.predict(X_test_scaled)

    best_rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    best_r2_test = r2_score(y_test, y_test_pred)

    print(f"\nBest model Test RMSE: {best_rmse_test:.4f}")
    print(f"Best model Test R²  : {best_r2_test:.4f}")

    # how many sigma better than baseline in terms of RMSE
    delta_rmse = baseline_rmse_test - best_rmse_mean
    if best_rmse_std > 0.0:
        sigma_units = delta_rmse / best_rmse_std
    else:
        sigma_units = np.nan

    print(
        f"\nDifference between baseline and best model mean CV RMSE "
        f"(in units of best model's CV RMSE std): {sigma_units:.2f} σ"
    )

    # save test predictions
    test_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_test_pred,
            "residual": y_test_pred - y_test,
        }
    )
    test_pred_path = RESULTS / "test_predictions_best_model.csv"
    test_df.to_csv(test_pred_path, index=False)
    print(f"Saved test predictions to: {test_pred_path}")

    # save txt summary of baseline vs best model
    summary_txt_path = RESULTS / "baseline_vs_best.txt"
    with open(summary_txt_path, "w") as f:
        f.write("Baseline vs Best Model (Test and CV)\n")
        f.write("------------------------------------\n")
        f.write(f"Baseline constant prediction (mean y_other): {baseline_mean:.6f}\n")
        f.write(f"Baseline Test RMSE: {baseline_rmse_test:.6f}\n")
        f.write(f"Baseline Test R²: {baseline_r2_test:.6f}\n")
        f.write("\n")
        f.write(f"Best model name: {best_model_name}\n")
        f.write(f"Best model CV RMSE mean: {best_rmse_mean:.6f}\n")
        f.write(f"Best model CV RMSE std : {best_rmse_std:.6f}\n")
        f.write(f"Best model Test RMSE   : {best_rmse_test:.6f}\n")
        f.write(f"Best model Test R²     : {best_r2_test:.6f}\n")
        f.write("\n")
        f.write(
            "Improvement of best model over baseline in units of best model "
            f"CV RMSE std: {sigma_units:.4f} σ\n"
        )

    print(f"Saved baseline vs best model summary to: {summary_txt_path}")


def main():
    run_cv_and_test()


if __name__ == "__main__":
    main()