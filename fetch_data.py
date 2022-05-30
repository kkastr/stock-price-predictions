import os

import argparse
import datetime
import pandas as pd
from enum import Enum
import pandas_datareader as pweb

parser = argparse.ArgumentParser(description="""Download data from yahoo finance.""")

parser.add_argument(
    "--update", dest="update", action="store_true", default=False, help="Update data file"
)
parser.add_argument(
    "-ticker", dest="ticker", type=str, help="The ticker for the security (i.e. GOOG, AMZN, etc)."
)
parser.add_argument(
    "-from",
    dest="start_date",
    type=datetime.date.fromisoformat,
    help="The first date in the period of interest.",
)
parser.add_argument(
    "-to",
    dest="end_date",
    type=datetime.date.fromisoformat,
    help="The last date in the period of interest.",
)


class Interval(Enum):
    days = 1
    months = 12
    years = 365

    @classmethod
    def fmt_str(cls, day_dist):
        try:
            assert day_dist > 0
        except AssertionError:
            print("Negative number of days. Make sure you have entered the dates correctly.")
            return

        if day_dist <= 1:
            return f"{day_dist}d"
        elif day_dist < 365:
            _res = day_dist // cls.months.value
            return f"{_res}M"
        elif day_dist >= 365:
            _res = day_dist // cls.years.value
            return f"{_res}Y"


def getData(ticker, start_date, end_date, update):

    if not os.path.isdir("./data/"):
        os.system("mkdir -p ./data/")

    if not os.path.isdir("./plots/"):
        os.system("mkdir -p ./plots/")

    duration = end_date - start_date

    period = Interval.fmt_str(duration.days)

    filename = f"{period}_{ticker}_stock_prices_.csv"

    if update or not os.path.isfile(f"./data/{filename}"):

        temp = pweb.DataReader([ticker], "yahoo", start_date, end_date)

        temp = temp.T.unstack("Symbols").T
        temp.columns = temp.columns.get_level_values("Attributes").values
        temp["Ticker"] = temp.index.get_level_values("Symbols")
        temp["Date"] = temp.index.get_level_values("Date")

        temp.to_csv(f"./data/{filename}", index=False)

        print("Downloaded data.")
    else:
        print("Data found.")


if __name__ == "__main__":
    args = parser.parse_args()

    getData(**vars(args))
