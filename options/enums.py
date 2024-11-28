from enums import StrEnum


class ProblemType(StrEnum):

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelNamesClassification(StrEnum):

    LOGISTIC_REGRESSION = "logistic_regression"
    K_NEIGHBORS_CLASSIFIER = "k_neighbors_classifier"
    SVC = "svc"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    ADA_BOOST_CLASSIFIER = "ada_boost_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    XGB_CLASSIFIER = "xgb_classifier"
    CAT_BOOST_CLASSIFIER = "cat_boost_classifier"
    LGBM_CLASSIFIER = "lgbm_classifier"
    MLP_CLASSIFIER = "mlp_classifier"


class ModelNamesRegression(StrEnum):

    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    K_NEIGHBORS_REGRESSOR = "k_neighbors_regressor"
    SVR = "svr"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    ADA_BOOST_REGRESSOR = "ada_boost_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    XGB_REGRESSOR = "xgb_regressor"
    CAT_BOOST_REGRESSOR = "cat_boost_regressor"
    LGBM_REGRESSOR = "lgbm_regressor"
    MLP_REGRESSOR = "mlp_regressor"


class ImputationMethodType(StrEnum):

    IMPUTATION_DICTIONARY = "imputation_dictionary"
    GLOBAL_IMPUTATION = "global_imputation"


class ImputationMethod(StrEnum):

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FILL_NA = "fillna"
