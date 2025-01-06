def get_params(trial, model_name):
    params = {
        "LGBMClassifier": {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-9, 100.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-9, 100.0),
            "random_state": 42,
        },
        "RandomForestClassifier": {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
            "random_state": 42,
        },
        "CatBoostClassifier": {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
            "depth": trial.suggest_int("depth", 3, 20),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-9, 100.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_state": 42,
        },
    }

    return params[model_name]


from skopt.space import Real, Integer, Categorical

param_space = {
    "LGBMClassifier": {
        "model__n_estimators": Integer(100, 1000),
        "model__max_depth": Integer(3, 20),
        "model__min_child_samples": Integer(31, 100),
        "model__reg_alpha": Real(5, 100.0),
        "model__reg_lambda": Real(5, 100.0),
        # "model__num_leaves": Integer(10, 100),
        # "model__learning_rate": Real(1e-8, 1.0),
        # "model__subsample": Real(0.1, 1.0),
        # "model__colsample_bytree": Real(0.1, 1.0),
    },
    "RandomForestClassifier": {
        "model__n_estimators": Integer(100, 1000),
        "model__max_depth": Integer(3, 7),
        "model__min_samples_split": Integer(2, 20),
        "model__min_samples_leaf": Integer(10, 20),
        "model__max_features": Real(0.1, 0.7),
    },
    "CatBoostClassifier": {
        "model__iterations": Integer(50, 300),
        "model__learning_rate": Real(0.001, 0.01),
        "model__depth": Integer(3, 10),
        "model__l2_leaf_reg": Real(1, 100.0),
        "model__border_count": Integer(32, 255),
    },
    "XGBClassifier": {
        "model__n_estimators": Integer(50, 300),
        "model__max_depth": Integer(3, 20),
        "model__learning_rate": Real(1e-8, 1.0),
        "model__colsample_bytree": Real(0.1, 0.7),
        "model__reg_alpha": Real(1, 100.0),
        "model__reg_lambda": Real(1, 100.0),
    },
}
