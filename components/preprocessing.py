import pandas as pd


class DataManipulation:

    def __init__(self, data, opt, logger) -> None:
        self.data = data
        self._opt = opt
        self._logger = logger

    def change_column_dtype(self):

        dtype_dict = self._opt.dtype_dict

        if dtype_dict == ["None"] or dtype_dict is None:
            self._logger.warning("No data type change is mentioned or required.")
            self._logger.info(
                "Check the data types of the columns or training options.py file."
            )
            return

        for column, dtype in dtype_dict.items():
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' does not exist in the data.")
            try:
                if dtype == "datetime64":
                    self.data[column] = pd.to_datetime(self.data[column])
                else:
                    self.data[column] = self.data[column].astype(dtype)
                self._logger.info(
                    f"Column '{column}' has been converted to '{dtype}' type"
                )

            except Exception as e:
                raise ValueError(
                    f"Failed to convert column '{column}' to '{dtype}': {e}"
                )

    def impute_missing_values(self):

        if not self.data.isnull().values.any():
            self._logger.info("No missing values in the dataset.")
            self._logger.info("Skipping the missing value imputation step.")

        method_type = self._opt.missing_values_imputation_method

        if method_type == "imputation_dictionary":
            self._logger.info(
                "Imputing missing values using imputation dictionary method."
            )

            imputation_dict = self._opt.missing_values_imputation

            try:
                for column, (method, fill_value) in imputation_dict.items():
                    if method == "mean":
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                    elif method == "median":
                        self.data[column].fillna(
                            self.data[column].median(), inplace=True
                        )
                    elif method == "mode":
                        self.data[column].fillna(
                            self.data[column].mode()[0], inplace=True
                        )
                    elif method == "fillna":
                        if fill_value is None:
                            raise ValueError(
                                "fill_value must be provided when using 'fillna' method."
                            )
                        self.data[column].fillna(fill_value, inplace=True)
                    else:
                        raise ValueError(
                            "Invalid imputation method. Choose from 'mean', 'median', 'mode', 'fillna'."
                        )
            except Exception as e:
                raise ValueError(f"Error occurred during missing value imputation: {e}")

        elif method_type == "global_imputation":

            self._logger.info("Imputing missing values using global imputation method.")

            try:
                if self._opt.global_imputation_method == "mean":
                    self.data.fillna(self.data.mean(), inplace=True)
                elif self._opt.global_imputation_method == "median":
                    self.data.fillna(self.data.median(), inplace=True)
                elif self._opt.global_imputation_method == "mode":
                    self.data.fillna(self.data.mode().iloc[0], inplace=True)
                elif self._opt.global_imputation_method == "fillna":
                    if self._opt.global_fill_value is None:
                        raise ValueError(
                            "fill_value must be provided when using 'fillna' method."
                        )
                    self.data.fillna(self._opt.global_fill_value, inplace=True)
                else:
                    raise ValueError(
                        "Invalid imputation method. Choose from 'mean', 'median', 'mode', 'fillna'."
                    )

                self._logger.info(
                    f"Imputed missing values using '{self._opt.global_imputation_method}' method."
                )

            except Exception as e:
                raise ValueError(f"Error occurred during missing value imputation: {e}")

        else:
            raise ValueError(
                "Invalid imputation method type. Choose from 'imputation_dictionary', 'global_imputation'."
            )

    def multivariate_impute(self):
        """
        Perform multivariate imputation based on custom conditions.
        Placeholder for future custom logic.
        """
        # Template for multivariate imputation based on custom conditions.
        # Add the conditions here.
        pass

    def feature_engineering(self):

        if self._opt.calculate_feature_engg:
            if (
                not isinstance(self._opt.feature_engg_names, list)
                or not self._opt.feature_engg_names
            ):
                raise ValueError("Feature engineering names must be a non-empty list.")

            try:
                from components.call_methods import call_feature_engineering

                for feature_engg_name in self._opt.feature_engg_names:

                    feature_engineering_func = call_feature_engineering(
                        feature_engg_name, self.data, self.logger, self._opt
                    )

                    self._logger.info(
                        f"Applying feature engineering: {feature_engg_name}"
                    )
                    self.data = feature_engineering_func()
                    self._logger.info(
                        f"Feature engineering '{feature_engg_name}' applied successfully."
                    )

            except Exception as e:
                error_message = f"Error occurred during feature engineering '{feature_engg_name}': {e}"
                self.logger.error(error_message)
                raise ValueError(error_message)

        else:
            self._logger.info("Skipping feature engineering step.")
            self._logger.info(
                "Set'calculate_feature_engg' to True in training options.py to perform feature engineering."
            )

    def encode_columns(self):

        # ----- Label Encoding -----

        if self._opt.do_label_encode:
            try:
                label_encode_cols = self._opt.label_encode_columns
                from sklearn.preprocessing import LabelEncoder

                for column in label_encode_cols:
                    if column not in self.data.columns:
                        raise KeyError(
                            f"Column '{column}' not found for label encoding."
                        )

                    le = LabelEncoder()
                    self.data[column] = le.fit_transform(self.data[column])
                    self._logger.info(f"Column '{column}' has been label encoded.")

            except Exception as e:
                self._logger.error(f"Unexpected error during label encoding: {e}")
                raise ValueError(f"Unexpected error during label encoding: {e}")

        else:
            self._logger.info("No label encoding to be performed.")
            self._logger.info(
                "Set 'do_label_encode' to True in the configuration to perform label encoding."
            )

        # ----- One-Hot Encoding -----

        if self._opt.do_one_hot_encode:
            try:
                one_hot_encode_cols = self._opt.one_hot_encode_columns
                for column in one_hot_encode_cols:
                    if column not in self.data.columns:
                        raise KeyError(
                            f"Column '{column}' not found for one-hot encoding."
                        )

                # Perform one-hot encoding with pandas
                self.data = pd.get_dummies(
                    self.data, columns=one_hot_encode_cols, drop_first=True
                )

                self._logger.info(
                    f"Columns {one_hot_encode_cols} have been one-hot encoded."
                )

                self._logger.info(f"Data after one-hot encoding:\n{self.data.head()}")

            except Exception as e:
                self._logger.error(f"Unexpected error during one-hot encoding: {e}")
                raise ValueError(f"Unexpected error during one-hot encoding: {e}")

        else:
            self._logger.info("No one-hot encoding to be performed.")
            self._logger.info(
                "Set 'do_one_hot_encode' to True in the configuration to perform one-hot encoding."
            )

        return self.data

    def drop_columns(self):

        if self._opt.drop_columns_condition:
            self._logger.info("Dropping columns is set to True in the configuration.")

            if (
                not isinstance(self._opt.drop_columns, list)
                or not self._opt.drop_columns
            ):
                self._logger.warning(
                    "Specified column is not in the list. Skipping this step."
                )
                return

            try:
                missing_columns = [
                    col
                    for col in self._opt.drop_columns
                    if col not in self.data.columns
                ]
                if missing_columns:
                    raise ValueError(
                        f"The following columns are missing and cannot be dropped: {missing_columns}"
                    )

                # Drop specified columns
                self.data.drop(columns=self._opt.drop_columns, inplace=True)
                self._logger.info(
                    f"Dropped columns: {self._opt.drop_columns} from the dataset."
                )

                return self.data

            except Exception as e:
                error_message = f"Unexpected error during column dropping: {e}"
                self._logger.error(error_message)
                raise ValueError(error_message)

        else:
            self._logger.info("Skipping the dropping columns step.")
            self._logger.info(
                "Set 'drop_columns_condition' to True in the configuration to drop columns."
            )
            return self.data

    def run_preprocessing(self):
        self.change_column_dtype()
        self.impute_missing_values()
        self.feature_engineering()
        self.encode_columns()
        self.drop_columns()
        self._logger.info("Data preprocessing completed successfully.")
        return self.data
