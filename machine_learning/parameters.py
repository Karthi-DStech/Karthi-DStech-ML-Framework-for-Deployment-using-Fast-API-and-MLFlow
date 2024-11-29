from options.enums import ModelNamesClassification, ModelNamesRegression


class ClassificationHyperparameters:
    """
    This class defines hyperparameters for different classification models.

    Parameters
    ----------
    None

    Attributes
    ----------
    _param_name : str
        The name of the hyperparameters to create.

    """

    def __init__(self):
        self._param_name = None

    def get_logistic_regression_params(self, trial):
        """
        Define and return hyperparameters for Logistic Regression.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Logistic Regression.
        """
        self._param_name = ModelNamesClassification.LOGISTIC_REGRESSION
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
        }
        return params

    def get_knn_classifier_params(self, trial):
        """
        Define and return hyperparameters for K-Nearest Neighbors.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for K-Nearest Neighbors.
        """
        self._param_name = ModelNamesClassification.K_NEIGHBORS_CLASSIFIER
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 50),
            "p": trial.suggest_int("p", 1, 2),
        }
        return params

    def get_svc_params(self, trial):
        """
        Define and return hyperparameters for Support Vector Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Support Vector Classifier.
        """
        self._param_name = ModelNamesClassification.SVC
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "degree": trial.suggest_int("degree", 2, 5),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
        }
        return params

    def get_decision_tree_classifier_params(self, trial):
        """
        Define and return hyperparameters for Decision Tree Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Decision Tree Classifier.
        """
        self._param_name = ModelNamesClassification.DECISION_TREE_CLASSIFIER
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        return params

    def get_random_forest_classifier_params(self, trial):
        """
        Define and return hyperparameters for Random Forest Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Random Forest Classifier.
        """

        self._param_name = ModelNamesClassification.RANDOM_FOREST_CLASSIFIER
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]
            ),
        }
        return params

    def get_ada_boost_classifier_params(self, trial):
        """
        Define and return hyperparameters for AdaBoost Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for AdaBoost Classifier.
        """
        self._param_name = ModelNamesClassification.ADA_BOOST_CLASSIFIER
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"]),
        }
        return params

    def get_gradient_boosting_classifier_params(self, trial):
        """
        Define and return hyperparameters for Gradient Boosting Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Gradient Boosting Classifier.
        """
        self._param_name = ModelNamesClassification.GRADIENT_BOOSTING_CLASSIFIER
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        return params

    def get_xgboost_classifier_params(self, trial):
        """
        Define and return hyperparameters for XGBoost Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for XGBoost Classifier.
        """
        self._param_name = ModelNamesClassification.XGB_CLASSIFIER
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }
        return params

    def get_cat_boost_classifier_params(self, trial):
        """
        Define and return hyperparameters for CatBoost Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for CatBoost Classifier.
        """
        self._param_name = ModelNamesClassification.CAT_BOOST_CLASSIFIER
        params = {
            "iterations": trial.suggest_int("iterations", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "verbose": 0,
        }
        return params

    def get_light_gbm_classifier_params(self, trial):
        """
        Define and return hyperparameters for LightGBM Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for LightGBM Classifier.
        """
        self._param_name = ModelNamesClassification.LGBM_CLASSIFIER
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 31, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "verbose": -1,
        }
        return params


class RegressionHyperparameters:
    """
    This class defines hyperparameters for different regression models.

    Attributes
    ----------
    _param_name : str
        The name of the hyperparameters to create.
    """

    def __init__(self):
        self._param_name = None

    def get_linear_regression_params(self, trial):
        """
        Define and return hyperparameters for Linear Regression.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Linear Regression.
        """
        self._param_name = ModelNamesRegression.LINEAR_REGRESSION
        params = {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "normalize": trial.suggest_categorical("normalize", [True, False]),
        }
        return params

    def get_ridge_params(self, trial):
        """
        Define and return hyperparameters for Ridge Regression.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for Ridge Regression.
        """
        self._param_name = ModelNamesRegression.RIDGE
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
            ),
        }
        return params

    def get_lasso_params(self, trial):
        """
        Define and return hyperparameters for Lasso Regression.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for Lasso Regression.
        """
        self._param_name = ModelNamesRegression.LASSO
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
        }
        return params

    def get_knn_regressor_params(self, trial):
        """
        Define and return hyperparameters for K-Nearest Neighbors Regressor.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for K-Nearest Neighbors Regressor.
        """
        self._param_name = ModelNamesRegression.K_NEIGHBORS_REGRESSOR
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
            "weights": trial.suggest_categorical(["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 50),
        }
        return params

    def get_svr_params(self, trial):
        """
        Define and return hyperparameters for Support Vector Regressor (SVR).

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for SVR.
        """
        self._param_name = ModelNamesRegression.SVR
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "degree": trial.suggest_int("degree", 2, 5),
            "gamma": trial.suggest_categorical(["scale", "auto"]),
        }
        return params

    def get_decision_tree_regressor_params(self, trial):
        """
        Define and return hyperparameters for Decision Tree Regressor.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for Decision Tree Regressor.
        """
        self._param_name = ModelNamesRegression.DECISION_TREE_REGRESSOR
        params = {
            "criterion": trial.suggest_categorical(
                "criterion",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            ),
            "splitter": trial.suggest_categorical(["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        return params

    def get_random_forest_regressor_params(self, trial):
        """
        Define and return hyperparameters for Random Forest Regressor.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for Random Forest Regressor.
        """
        self._param_name = ModelNamesRegression.RANDOM_FOREST_REGRESSOR
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "criterion": trial.suggest_categorical(
                "criterion",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
        return params

    def get_gradient_boosting_regressor_params(self, trial):
        """
        Define and return hyperparameters for Gradient Boosting Regressor.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for Gradient Boosting Regressor.
        """
        self._param_name = ModelNamesRegression.GRADIENT_BOOSTING_REGRESSOR
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        return params

    def get_xgb_regressor_params(self, trial):
        """
        Define and return hyperparameters for XGBoost Regressor.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for XGBoost Regressor.
        """
        self._param_name = ModelNamesRegression.XGB_REGRESSOR
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }
        return params

    def get_cat_boost_regressor_params(self, trial):
        """
        Define and return hyperparameters for CatBoost Regressor.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for CatBoost Regressor.
        """
        self._param_name = ModelNamesRegression.CAT_BOOST_REGRESSOR
        params = {
            "iterations": trial.suggest_int("iterations", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "verbose": 0,
        }
        return params

    def get_lgbm_regressor_params(self, trial):
        """
        Define and return hyperparameters for LightGBM Regressor.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict
            A dictionary of hyperparameters for LightGBM Regressor.
        """
        self._param_name = ModelNamesRegression.LGBM_REGRESSOR
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 31, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "verbose": 0,
        }
        return params
