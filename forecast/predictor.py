import datetime
from typing import Literal

import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from numba import njit
from sklearn.ensemble import RandomForestRegressor
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean
from xgboost import XGBRegressor


class DataReader:
    def __init__(self, file_path: str):
        data = pd.read_csv(file_path)

        data["CNES"] = data["CNES"].astype(str).str.zfill(7)
        data["DT"] = pd.to_datetime(data["DT"])

        self.timeseries = data

    def cut_data(self, cut_point: datetime.date) -> pd.DataFrame:
        return self.timeseries.loc[self.timeseries["DT"] < np.datetime64(cut_point)]


class Predictor:
    def __init__(self, freq: Literal["D", "W", "M"] = "W"):
        self.model = MLForecast(
            [
                XGBRegressor(),
                RandomForestRegressor(random_state=0),
            ],
            freq,
            lags=[1, 2],
            lag_transforms={1: [expanding_mean], 4: [self.rolling_mean_4]},
            target_transforms=[Differences([1])],
        )

    def fit(self, data: pd.DataFrame):
        self.model.fit(data, "CNES", "DT", "HOSPITAL")

    def forecast(
        self,
        data_reader: DataReader,
        start_date: datetime.date = datetime.date.today(),
        end_date: datetime.date | None = None,
    ):
        self.fit(data_reader.cut_data(start_date))

        h = (end_date - start_date).days if end_date is not None else 1

        predictions = self.model.predict(h)

        return predictions

    def forecast_h(
        self,
        data_reader: DataReader,
        start_date: datetime.date = datetime.date.today(),
        h: int = 1,
    ):
        self.fit(data_reader.cut_data(start_date))

        predictions = self.model.predict(h)

        return predictions

    @staticmethod
    @njit
    def rolling_mean_4(x):
        return rolling_mean(x, window_size=4)


def main():
    reader = DataReader("forecast/timeseries_semanal.csv")
    model = Predictor()

    model.fit(reader.timeseries)

    # prediction = model.forecast(
    #     reader,
    #     datetime.date(2024, 1, 1),
    #     datetime.date.today(),
    # )
    prediction = model.forecast_h(
        reader,
        datetime.date(2024, 1, 1),
        4,
    )
    return prediction


if __name__ == "__main__":
    print(main())
