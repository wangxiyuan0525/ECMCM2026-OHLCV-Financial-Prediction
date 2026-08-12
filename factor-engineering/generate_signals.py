import numpy as np

def generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray:
    """Generate Ret5 and Ret60 prediction signals.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier (e.g. "dataset0").
    factors : np.ndarray
        Shape (T, F) float32 factor matrix from factor.py.

    Returns
    -------
    signals : np.ndarray
        Shape (T, 2) float32.  Column 0 = Ret5 signal, Column 1 = Ret60 signal.
    """
    import sys
    from pathlib import Path

    import joblib
    import numpy as np

    MODEL_PATH = Path("/workspace/submission/models_ret5_ret60.joblib")
    FEATURE_CLIP_VALUE = 1e6
    PRED_CLIP_VALUE = 1e3

    def _stub_lgb_callbacks():
        main_module = sys.modules.get("__main__")
        if main_module is not None:
            setattr(main_module, "lgb_ic_objective",
                    lambda y_true, y_pred, weight=None: (
                        np.zeros(np.asarray(y_true).shape, dtype=np.float32),
                        np.ones(np.asarray(y_true).shape, dtype=np.float32),
                    ))
            setattr(main_module, "lgb_ic_eval",
                    lambda y_true, y_pred, weight=None: ("ic", 0.0, True))

    def _predict_one_model(model, X):
        best_iter = getattr(model, "ic_best_iteration_", None)
        if best_iter is None:
            best_iter = getattr(model, "best_iteration_", None)
        if isinstance(best_iter, (int, np.integer)) and best_iter > 0:
            try:
                return model.predict(X, num_iteration=int(best_iter))
            except TypeError:
                return model.predict(X)
        return model.predict(X)

    def _predict_ensemble(models, X):
        preds = []
        for m in models:
            p = _predict_one_model(m, X)
            preds.append(p.astype(np.float64))
        return np.mean(preds, axis=0).astype(np.float32)

    _stub_lgb_callbacks()

    model_path = MODEL_PATH
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent / MODEL_PATH.name

    bundle = joblib.load(model_path)
    model_ret5 = bundle["ret5_model"]
    model_ret60 = bundle["ret60_model"]
    feature_count = int(bundle.get("feature_count") or 250)

    if factors.ndim != 2 or factors.shape[1] < feature_count:
        raise ValueError(
            f"bad factors shape: {factors.shape}, need at least {feature_count} columns"
        )

    X = np.asarray(factors[:, :feature_count], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)

    if isinstance(model_ret5, list):
        pred5 = _predict_ensemble(model_ret5, X)
    else:
        pred5 = _predict_one_model(model_ret5, X)

    if isinstance(model_ret60, list):
        pred60 = _predict_ensemble(model_ret60, X)
    else:
        pred60 = _predict_one_model(model_ret60, X)

    signals = np.stack([pred5, pred60], axis=1).astype(np.float32)
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    signals = np.clip(signals, -PRED_CLIP_VALUE, PRED_CLIP_VALUE)
    return signals.astype(np.float32, copy=False)
