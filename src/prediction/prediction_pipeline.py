from prediction.features.expected_shortfall import run as run_es
from prediction.features.obv_signal        import run as run_obv


def run_prediction_pipeline():
    print("=" * 50)
    print("PREDICTION PIPELINE")
    print("=" * 50)

    print("\n[1/2] Expected Shortfall (var_95, es_95, es_ratio)")
    run_es()

    print("\n[2/2] OBV Signal (obv_signal)")
    run_obv()

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    run_prediction_pipeline()
