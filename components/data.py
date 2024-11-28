import pandas as pd
from utils.errors import DataLoadingError


class DataBuilder:

    def __init__(self, file_path: str, opt, logger: object = None) -> None:
        self._file_path = file_path
        self._logger = logger
        self._opt = opt
        self.data = None

    def load_data(self):

        assert self._file_path.endswith(".csv"), "The file path must end with '.csv'."

        if self.data is None:
            try:
                self.data = pd.read_csv(self._file_path)
                self._logger.info(f"Data loaded from {self._file_path}")
                return self.data

            except FileNotFoundError:
                raise DataLoadingError(f"File not found: {self._file_path}")
            except pd.errors.ParserError as e:
                raise DataLoadingError(f"Error parsing CSV file: {e}")
            except Exception as e:
                raise DataLoadingError(f"An error occurred while loading data. {e}")

    def data_info(self):

        if self.data is not None:
            num_samples = self.data.shape[0]
            num_features = self.data.shape[1]
            self._logger.info(
                f"Number of samples: {num_samples} and number of features: {num_features}"
            )

        return num_samples, num_features

    def check_data_types(self):

        categorical_cols = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numeric_cols = self.data.select_dtypes(include=["number"]).columns.tolist()

        other_cols = self.data.select_dtypes(
            exclude=["number", "object", "category"]
        ).columns.tolist()

        self._logger.info(
            "Data types separated into categorical, numeric, and other columns"
        )

        self.logger.info(f"Categorical columns: {categorical_cols}")
        self.logger.info(f"Numeric columns: {numeric_cols}")
        self.logger.info(f"Other columns: {other_cols}")

        return categorical_cols, numeric_cols, other_cols

    def check_missing_values(self):

        if self.data is not None:
            try:
                missing_values_exist = self.data.isnull().values.any()

                if missing_values_exist:
                    self._logger.warning("Missing values exist in the dataset.")

                    missing_columns = self.data.columns[
                        self.data.isnull().any()
                    ].tolist()
                    self._logger.warning(
                        f"Columns with missing values: {', '.join(missing_columns)}"
                    )

                else:
                    self._logger.info("No missing values in the dataset.")

                return missing_columns if missing_values_exist else []

            except Exception as e:
                raise ValueError(
                    f"An error occurred while checking for missing values. {e}"
                )

        else:
            raise DataLoadingError(
                "An error occurred while checking for missing values. Data is None."
            )

    def run_data(self):

        self.load_data()
        self.data_info()
        self.check_data_types()
        self.check_missing_values()
