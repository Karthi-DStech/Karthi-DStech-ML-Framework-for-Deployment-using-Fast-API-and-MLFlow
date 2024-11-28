from options.enums import ProblemType, ModelNamesClassification, ModelNamesRegression
from machine_learning.parameters import ClassificationHyperparameters
from typing import Dict


def get_classification_models(problem_type: str) -> Dict:

    if problem_type.lower() != ProblemType.CLASSIFICATION:
        raise ValueError(
            f"Invalid problem type: {problem_type}. Expected 'classification'."
        )

    return {
        ModelNamesClassification.LOGISTIC_REGRESSION: "sklearn.linear_model.LogisticRegression",
        ModelNamesClassification.K_NEIGHBORS_CLASSIFIER: "sklearn.neighbors.KNeighborsClassifier",
        ModelNamesClassification.SVC: "sklearn.svm.SVC",
        ModelNamesClassification.DECISION_TREE_CLASSIFIER: "sklearn.tree.DecisionTreeClassifier",
        ModelNamesClassification.RANDOM_FOREST_CLASSIFIER: "sklearn.ensemble.RandomForestClassifier",
        ModelNamesClassification.ADA_BOOST_CLASSIFIER: "sklearn.ensemble.AdaBoostClassifier",
        ModelNamesClassification.GRADIENT_BOOSTING_CLASSIFIER: "sklearn.ensemble.GradientBoostingClassifier",
        ModelNamesClassification.XGB_CLASSIFIER: "xgboost.XGBClassifier",
        ModelNamesClassification.CAT_BOOST_CLASSIFIER: "catboost.CatBoostClassifier",
        ModelNamesClassification.LGBM_CLASSIFIER: "lightgbm.LGBMClassifier",
        ModelNamesClassification.MLP_CLASSIFIER: "sklearn.neural_network.MLPClassifier",
    }


def get_regression_models(problem_type: str) -> Dict:

    if problem_type.lower() != ProblemType.REGRESSION:
        raise ValueError(
            f"Invalid problem type: {problem_type}. Expected 'regression'."
        )

    return {
        ModelNamesRegression.LINEAR_REGRESSION: "sklearn.linear_model.LinearRegression",
        ModelNamesRegression.RIDGE: "sklearn.linear_model.Ridge",
        ModelNamesRegression.LASSO: "sklearn.linear_model.Lasso",
        ModelNamesRegression.K_NEIGHBORS_REGRESSOR: "sklearn.neighbors.KNeighborsRegressor",
        ModelNamesRegression.SVR: "sklearn.svm.SVR",
        ModelNamesRegression.DECISION_TREE_REGRESSOR: "sklearn.tree.DecisionTreeRegressor",
        ModelNamesRegression.RANDOM_FOREST_REGRESSOR: "sklearn.ensemble.RandomForestRegressor",
        ModelNamesRegression.ADA_BOOST_REGRESSOR: "sklearn.ensemble.AdaBoostRegressor",
        ModelNamesRegression.GRADIENT_BOOSTING_REGRESSOR: "sklearn.ensemble.GradientBoostingRegressor",
        ModelNamesRegression.XGB_REGRESSOR: "xgboost.XGBRegressor",
        ModelNamesRegression.CAT_BOOST_REGRESSOR: "catboost.CatBoostRegressor",
        ModelNamesRegression.LGBM_REGRESSOR: "lightgbm.LGBMRegressor",
        ModelNamesRegression.MLP_REGRESSOR: "sklearn.neural_network.MLPRegressor",
    }


def make_params(param_name: ModelNamesClassification, *args, **kwargs):

    # Mapping of Enum values to their respective hyperparameter methods

    hyperparams = ClassificationHyperparameters(*args, **kwargs)

    param_funcs = {
        # ------ Classification Model Hyperparameters ------
        ModelNamesClassification.LOGISTIC_REGRESSION: hyperparams.get_logistic_regression_params,
        ModelNamesClassification.K_NEIGHBORS_CLASSIFIER: hyperparams.get_knn_classifier_params,
        ModelNamesClassification.SVC: hyperparams.get_svc_params,
        ModelNamesClassification.DECISION_TREE_CLASSIFIER: hyperparams.get_decision_tree_classifier_params,
        ModelNamesClassification.RANDOM_FOREST_CLASSIFIER: hyperparams.get_random_forest_classifier_params,
        ModelNamesClassification.ADA_BOOST_CLASSIFIER: hyperparams.get_ada_boost_classifier_params,
        ModelNamesClassification.GRADIENT_BOOSTING_CLASSIFIER: hyperparams.get_gradient_boosting_classifier_params,
        ModelNamesClassification.XGB_CLASSIFIER: hyperparams.get_xgboost_classifier_params,
        ModelNamesClassification.CAT_BOOST_CLASSIFIER: hyperparams.get_cat_boost_classifier_params,
        ModelNamesClassification.LGBM_CLASSIFIER: hyperparams.get_light_gbm_classifier_params,
        # ------ Regression Model Hyperparameters ------
        ModelNamesRegression.LINEAR_REGRESSION: hyperparams.get_linear_regression_params,
        ModelNamesRegression.RIDGE: hyperparams.get_ridge_params,
        ModelNamesRegression.LASSO: hyperparams.get_lasso_params,
        ModelNamesRegression.K_NEIGHBORS_REGRESSOR: hyperparams.get_knn_regressor_params,
        ModelNamesRegression.SVR: hyperparams.get_svr_params,
        ModelNamesRegression.DECISION_TREE_REGRESSOR: hyperparams.get_decision_tree_regressor_params,
        ModelNamesRegression.RANDOM_FOREST_REGRESSOR: hyperparams.get_random_forest_regressor_params,
        ModelNamesRegression.ADA_BOOST_REGRESSOR: hyperparams.get_ada_boost_regressor_params,
        ModelNamesRegression.GRADIENT_BOOSTING_REGRESSOR: hyperparams.get_gradient_boosting_regressor_params,
        ModelNamesRegression.XGB_REGRESSOR: hyperparams.get_xgb_regressor_params,
        ModelNamesRegression.CAT_BOOST_REGRESSOR: hyperparams.get_cat_boost_regressor_params,
        ModelNamesRegression.LGBM_REGRESSOR: hyperparams.get_light_gbm_regressor_params,
    }

    # Return the corresponding function or raise an error
    try:
        return param_funcs[param_name]
    except KeyError:
        raise NotImplementedError(
            f"Hyperparameters for {param_name} are not implemented."
        )
