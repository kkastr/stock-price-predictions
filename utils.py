import os
import argparse
import datetime
import pandas as pd
from enum import Enum
import pandas_datareader as pweb

parser = argparse.ArgumentParser(description="""Download data from yahoo finance.""")

parser.add_argument(
    "--update", dest="update", action="store_true", default=False, help="Update data (Overwrite)."
)

parser.add_argument(
    "--download", dest="download", action="store_true", default=False, help="Download data."
)
parser.add_argument(
    "-tickers",
    dest="tickers",
    type=str,
    nargs="+",
    help="The tickers for the securities (example: --tickers GOOG AMZN MSFT).",
)
parser.add_argument(
    "-from",
    dest="start_date",
    type=datetime.date.fromisoformat,
    help="The first date in the period of interest (format: YYYY-MM-DD).",
)
parser.add_argument(
    "-to",
    dest="end_date",
    type=datetime.date.fromisoformat,
    help="The last date in the period of interest (format: YYYY-MM-DD).",
)


class DateShortName(Enum):
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


def fmt_path_strings(path_strings):

    ret = {}
    for pstr in path_strings:
        fmted = pstr.split("/")[2].split(".")[0].split("_")
        ticker = f"{fmted[1]}"
        period = f"{fmted[0]}"

        ret[ticker].append(period)
    return ret


def fmt_dataframe(df):

    df = df.T.unstack("Symbols").T
    df.columns = df.columns.get_level_values("Attributes").values
    df["Ticker"] = df.index.get_level_values("Symbols")
    df["Date"] = df.index.get_level_values("Date")

    df.reset_index(drop=True, inplace=True)

    return df


def getData(tickers, start_date, end_date, download, update):

    if not os.path.isdir("./data/"):
        os.system("mkdir -p ./data/")

    fname = "stocks.csv"
    data_exists = os.path.isfile(f"./data/{fname}")

    if download and not data_exists:

        temp = fmt_dataframe(pweb.DataReader(tickers, "yahoo", start_date, end_date))

        temp.to_csv(f"./data/{fname}", index=False)

        print("Downloaded data.")
    elif download and data_exists and update:

        # this is a destructive update. Tt will update all values (incl. dates).

        df = pd.read_csv(f"./data/{fname}")

        temp = fmt_dataframe(pweb.DataReader(tickers, "yahoo", start_date, end_date))

        newdf = pd.concat([df, temp], ignore_index=True)
        newdf.to_csv(f"./data/{fname}", index=False)

    elif download and data_exists and not update:
        df = pd.read_csv(f"./data/{fname}")

        keys = df.Ticker.unique()

        temp = fmt_dataframe(pweb.DataReader(tickers, "yahoo", start_date, end_date))

        for k in keys:
            temp.drop(temp[temp["Ticker"] == k].index, inplace=True)

        if temp.empty:
            print(
                """No new tickers were added.
                If you meant to update existing tickers please use the --update option"""
            )
        else:
            newdf = pd.concat([df, temp], ignore_index=True)
            newdf.to_csv(f"./data/{fname}", index=False)
            print("New data downloaded. Existing ticker data did NOT get updated.")

    else:
        print("Data found.")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.download or args.update:
        getData(**vars(args))
    else:
        pass
