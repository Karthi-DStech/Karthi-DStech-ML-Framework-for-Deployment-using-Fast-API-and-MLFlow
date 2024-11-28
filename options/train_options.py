from options.base_options import BaseOptions
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainOptions(BaseOptions):
    """Train options"""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """Initialize train options"""
        BaseOptions.initialize(self)

        # Training Parameters

        self._parser.add_argument(
            "-m",
            "--model_name",
            type=str,
            default="XGBClassifier",
            choices=[
                "LogisticRegression",
                "KNeighborsClassifier",
                "SVC",
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "AdaBoostClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
                "CatBoostClassifier",
                "LGBMClassifier",
            ],
            help="Name of the model to train",
        )

        self._parser.add_argument(
            "--random_state",
            type=int,
            default=101,
            help="Seed for random state",
        )

        self._parser.add_argument(
            "--test_size",
            type=float,
            default=0.3,
            help="Size of the test set",
        )

        self._parser.add_argument(
            "--n_trials",
            type=int,
            default=100,
            choices=[50, 100, 150, 200],
            help="Number of trials for hyperparameter tuning using Optuna",
        )

        self._parser.add_argument(
            "-s",
            "--scale_data",
            type=bool,
            default=True,
            choices=[True, False],
            help="Whether to scale the data",
        )

        self._parser.add_argument(
            "--target_column",
            type=str,
            default="Status",
            help="Name of the target column",
        )

        self._parser.add_argument(
            "--drop_columns_condition",
            type=bool,
            default=True,
            choices=[True, False],
            help="Whether to drop columns based on a condition",
        )

        self._parser.add_argument(
            "--drop_columns",
            type=list,
            default=["Age", "ID"],
            help="List of columns to drop",
        )

        # ---- Encoding Columns Parameters ----

        # -> Label Encoding

        self._parser.add_argument(
            "--do_label_encode",
            type=bool,
            default=True,
            choices=[True, False],
            help="Whether to perform label encoding",
        )

        self._parser.add_argument(
            "--label_encode_columns",
            type=list,
            default=["Sex", "Edema", "Status"],
            help="List of columns to label encode",
        )

        # -> One Hot Encoding

        self._parser.add_argument(
            "--do_one_hot_encode",
            type=bool,
            default=True,
            choices=[True, False],
            help="Whether to perform label encoding",
        )

        self._parser.add_argument(
            "--one_hot_encode_columns",
            type=list,
            default=["Drug", "Hepatomegaly", "Spiders", "Ascites"],
            help="List of columns to one hot encode",
        )

        # ---- Changing Data Types Parameters ----

        self._parser.add_argument(
            "--dtype_dict",
            type=dict,
            default=None,
            help="Dictionary of column names and their respective data types. \
                eg: {'Date of Admission': 'datetime64', 'Discharge Date': 'datetime64'}.",
        )

        # ---- Missing Value Imputation Parameters ----

        # -> Type of Imputation

        self._parser.add_argument(
            "--missing_values_imputation_method",
            type=str,
            default="imputation_dictionary",
            choices=["imputation_dictionary", "global_imputation"],
            help="Method to impute missing values",
        )

        # -> global_imputation
        # (Impute all missing values with a single model of value)

        self._parser.add_argument(
            "--global_imputation_method",
            choices=["mean", "median", "mode", "fillna"],
            default="fillna",
            help="Method to use for global imputation: 'mean', 'median', 'mode', or 'fillna'.",
        )

        self._parser.add_argument(
            "--global_fill_value",
            # ---- Change to str if you want to impute with string -----
            type=int,
            default=-1,
            help="Value to impute for global imputation. If you want to impute with string, change the type to str.",
        )

        # -> imputation_dictionary
        # (Impute missing values based on a dictionary)

        self._parser.add_argument(
            "--missing_values_imputation",
            type=dict,
            default={
                "Drug": ("fillna", "Unknown"),
                "Ascites": ("fillna", "Unknown"),
                "Hepatomegaly": ("fillna", "Unknown"),
                "Spiders": ("fillna", "Unknown"),
                "Cholesterol": ("mean", None),
                "Albumin": ("mean", None),
                "Copper": ("mean", None),
                "Alk_Phos": ("mean", None),
                "SGOT": ("mean", None),
                "Tryglicerides": ("mean", None),
                "Platelets": ("mean", None),
                "Prothrombin": ("mean", None),
                "Stage": ("mode", None),
            },
            help="Missing Value Imputation Dictionary. Key is the column name and \
            value is a tuple of the imputation method and the value to impute. \
            The imputation methods are mean, median and mode. \
            For mean, median and mode, the value to impute should be None. \
            For eg:{'Ever_Married': ('mode', None), Var_1: ('fillna', 'Unknown'),}.",
        )

        # ---- Feature Engineering Parameters ----

        self._parser.add_argument(
            "--calculate_feature_engg",
            type=bool,
            default=False,
            choices=[True, False],
            help="Whether to perform feature engineering",
        )

        self._parser.add_argument(
            "--feature_engg_name",
            type=str,
            default="calculate_total_days",
            choices=["calculate_total_days", "separate_date_columns"],
            help="Name of the feature engineering column. eg: ['calculate_total_days'].",
        )

        # -> calculate_total_days columns

        self._parser.add_argument(
            "--starting_date_ctd",
            type=str,
            default="Date of Admission",
            help="Starting date column for total days calculation using calculate_total_days in Feature Engineering.",
        )

        self._parser.add_argument(
            "--ending_date_ctd",
            type=str,
            default="Discharge Date",
            help="Ending date column for total days calculation",
        )

        # -> separate_date_columns columns

        self._parser.add_argument(
            "--date_column_sdc",
            type=str,
            default="Time",
            help="Date column to separate using separate_date_columns in Feature Engineering.",
        )
