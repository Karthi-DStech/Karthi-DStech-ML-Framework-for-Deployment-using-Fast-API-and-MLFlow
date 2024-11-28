import pandas as pd


class FeatureEngineeringLogics:
    def __init__(self, data, logger, opt):
        self.data = data
        self._logger = logger
        self._opt = opt

    def calculate_total_days(self):

        self.name = "calculate_total_days"

        try:

            date_admission_col = self._opt.starting_date_ctd
            discharge_date_col = self._opt.ending_date_ctd

            if (
                date_admission_col in self.data.columns
                and discharge_date_col in self.data.columns
            ):
                self.data["Total Days Admitted"] = (
                    pd.to_datetime(self.data[discharge_date_col])
                    - pd.to_datetime(self.data[date_admission_col])
                ).dt.days

                self._logger.info("Total Days Admitted feature created.")

            else:
                self._logger.warning(
                    "Date columns not found. Please check the column names."
                )

            return self.data
        except Exception as e:
            raise ValueError(f"Error occurred during total days calculation: {e}")

    def separate_date_columns(self):

        self.name = "separate_date_columns"

        try:

            date_column = self._opt.date_column_sdc

            # Extracting the date-time components conditionally

            if date_column in self.data.columns:

                if self.data[date_column].dt.year.isnull().sum() == 0:
                    self.data["Year"] = self.data[date_column].dt.year

                if self.data[date_column].dt.month.isnull().sum() == 0:
                    self.data["Month"] = self.data[date_column].dt.month

                if self.data[date_column].dt.day.isnull().sum() == 0:
                    self.data["Day"] = self.data[date_column].dt.day

                if self.data[date_column].dt.hour.isnull().sum() == 0:
                    self.data["Hour"] = self.data[date_column].dt.hour

                self.data["Day of Week"] = self.data[date_column].dt.dayofweek

                if self.data[date_column].dt.minute.isnull().sum() == 0:
                    self.data["Minute"] = self.data[date_column].dt.minute

                self._logger.info("Date columns separated successfully.")

            else:
                self._logger.warning(
                    "Date column not found. Please check the column name."
                )

            return self.data
        except Exception as e:
            raise ValueError(f"Error occurred during separate date columns: {e}")
