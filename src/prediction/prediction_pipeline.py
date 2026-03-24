from expected_shortfall import run as run_es
from xgboost_risk      import run as run_risk
from xgboost_direction import run as run_direction


def run_prediction_pipeline():
    print("=" * 50)
    print("PREDICTION PIPELINE")
    print("=" * 50)

    print("\n[1/3] Expected Shortfall (var_95, es_95, es_ratio)")
    run_es()

    print("\n[2/3] XGBoost Risk Model")
    run_risk()

    print("\n[3/3] XGBoost Direction Model")
    run_direction()

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    run_prediction_pipeline()
