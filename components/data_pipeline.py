import pandas as pd
from sklearn.model_selection import train_test_split


class DataPipeline:
    def __init__(self, data, opt, logger) -> None:
        self._opt = opt
        self._logger = logger
        self.data = data

    def _scale_data(self):

        if self._opt.scale_data:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            features = self.data.drop(columns=[self._opt.target_column])
            scaled_features = scaler.fit_transform(features)
            self.data = pd.concat(
                [
                    pd.DataFrame(scaled_features, columns=features.columns),
                    self.data[self._opt.target_column],
                ],
                axis=1,
            )
            self._logger.info("Data scaled successfully.")

        else:
            self._logger.info("Data scaling skipped.")

    def _generate_data_splits(self):
        """
        Splits the data into training and testing sets.
        """
        try:
            X = self.data.drop(columns=[self._opt.target_column])
            y = self.data[self._opt.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._opt.test_size, random_state=self._opt.random_state
            )

            self._logger.info("Train-test split completed successfully.")
            self._logger.info(f"Training set: {X_train.shape[0]} samples.")
            self._logger.info(f"Testing set: {X_test.shape[0]} samples.")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            error_message = f"Error occurred during train-test split: {e}"
            self._logger.error(error_message)
            raise ValueError(error_message)

    def run_data_split_pipeline(self):

        self._scale_data()
        final_data = self._generate_data_splits()

        return final_data
