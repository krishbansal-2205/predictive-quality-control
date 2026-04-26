"""
main.py
-------
CLI pipeline runner for both FD001 and FD003.
Trains models, runs EWMA analysis, generates all outputs.
Run with: python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data_processing, spc, modeling, explainability, utils


def run_pipeline(dataset_name: str) -> None:
    """Run the full analysis pipeline for a given dataset.

    Args:
        dataset_name: Either ``'FD001'`` or ``'FD003'``.
    """
    print("\n" + "=" * 65)
    print(f"  RUNNING PIPELINE FOR {dataset_name}")
    fault_info = (
        "1 fault mode (HPC)"
        if dataset_name == "FD001"
        else "2 fault modes (HPC + HPT)"
    )
    print(f"  Operating Conditions: 1 | Fault Modes: {fault_info}")
    print("=" * 65)

    # ---------------------------------------------------------------- #
    # Step 1: Load Data                                                  #
    # ---------------------------------------------------------------- #
    print(f"\n[1/6] Loading {dataset_name} data...")
    try:
        train_df, test_df = data_processing.prepare_dataset(dataset_name)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # ---------------------------------------------------------------- #
    # Step 2: Select engines for analysis                                #
    # ---------------------------------------------------------------- #
    unique_engines = test_df["engine_id"].unique()
    engines_to_analyze = [unique_engines[0]]

    # For FD003, add a second engine to demonstrate the SPC blind spot
    if dataset_name == "FD003" and len(unique_engines) > 10:
        engines_to_analyze.append(unique_engines[10])

    # ---------------------------------------------------------------- #
    # Step 3: EWMA Analysis                                              #
    # ---------------------------------------------------------------- #
    print(f"\n[2/6] Running EWMA Analysis on sensor_12...")
    ewma_results = {}
    for eng_id in engines_to_analyze:
        df_eng = test_df[test_df["engine_id"] == eng_id].copy()
        try:
            result = spc.run_ewma_analysis(df_eng, sensor_col="sensor_12")
            ewma_results[eng_id] = result
            spc.plot_ewma_matplotlib(
                result,
                engine_id=eng_id,
                save_path=Path(
                    f"outputs/plots/{dataset_name}_ewma_engine_{eng_id}.png"
                ),
            )
            print(f"  Engine {eng_id} EWMA Breach: {result['breach_cycle']}")
        except Exception as e:
            print(f"  ERROR Engine {eng_id} EWMA: {e}")
            ewma_results[eng_id] = None

    # ---------------------------------------------------------------- #
    # Step 4: Train ML Model                                             #
    # ---------------------------------------------------------------- #
    print(f"\n[3/6] Training XGBoost Model for {dataset_name}...")
    try:
        X_train, y_train = modeling.prepare_features_targets(train_df)
        X_test, y_test = modeling.prepare_features_targets(test_df)
        model = modeling.train_model(X_train, y_train, dataset_name)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # ---------------------------------------------------------------- #
    # Step 5: Evaluate and Predict                                       #
    # ---------------------------------------------------------------- #
    print(f"\n[4/6] Evaluating Model...")
    try:
        metrics = modeling.evaluate_model(
            model,
            X_test,
            y_test,
            dataset_name,
            save_path=Path(
                f"outputs/reports/{dataset_name}_classification_report.txt"
            ),
        )
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\n[5/6] ML Predictions and Business Value...")
    for eng_id in engines_to_analyze:
        df_eng = test_df[test_df["engine_id"] == eng_id].copy()
        actual_failure = df_eng["cycle"].max() + df_eng["RUL"].iloc[-1]
        try:
            ml_warning, proba_series = modeling.predict_failure_start(
                model, df_eng
            )
            print(f"  Engine {eng_id} ML Warning: Cycle {ml_warning}")
        except Exception as e:
            print(f"  ERROR Engine {eng_id} ML: {e}")
            ml_warning = None

        ewma_breach = (
            ewma_results[eng_id]["breach_cycle"]
            if ewma_results.get(eng_id)
            else None
        )
        try:
            results = utils.calculate_business_value(
                eng_id, dataset_name, actual_failure, ewma_breach, ml_warning
            )
            report_str = utils.format_business_value_report(results)
            print(report_str)
            report_path = Path(
                f"outputs/reports/{dataset_name}_business_value.txt"
            )
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(report_str + "\n")
        except Exception as e:
            print(f"  ERROR business value: {e}")

    # ---------------------------------------------------------------- #
    # Step 6: SHAP                                                       #
    # ---------------------------------------------------------------- #
    print(f"\n[6/6] Generating SHAP Plot for {dataset_name}...")
    try:
        shap_values, X_sample = explainability.generate_shap_values(
            model, X_test
        )
        explainability.plot_shap_summary_matplotlib(
            shap_values,
            X_sample,
            dataset_name,
            save_path=Path(
                f"outputs/plots/{dataset_name}_shap_summary.png"
            ),
        )
    except Exception as e:
        print(f"  ERROR SHAP: {e}")

    print(f"\n  {dataset_name} pipeline complete.")


if __name__ == "__main__":
    utils.ensure_output_dirs()
    run_pipeline("FD001")
    run_pipeline("FD003")
    print("\n" + "=" * 65)
    print("  ALL PIPELINES COMPLETE")
    print("  Run: streamlit run app/streamlit_app.py")
    print("=" * 65)
