import sys
import torch
import argparse
import numpy as np
import pandas as pd
import sqlite3 as sql
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(
    description="""Train NN to predict stock prices and plot the result"""
)

parser.add_argument(
    "-ticker", dest="ticker", type=str, help="The ticker for the security (i.e. GOOG, AMZN, etc)."
)
parser.add_argument(
    "-period",
    dest="time_period",
    type=str,
    help="The period of time for the downloaded data: a number suffixed d, M, or Y.",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers

        self.lstm0 = nn.LSTMCell(1, self.hidden_layers, device=device)
        self.lstm1 = nn.LSTMCell(self.hidden_layers, self.hidden_layers, device=device)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.hidden_layers, 1, device=device)

    def forward(self, input_tensor, batch_size=1, future_preds=0):
        predictions = []
        ht0 = torch.zeros(batch_size, self.hidden_layers, dtype=torch.float32, device=device)
        ct0 = torch.zeros(batch_size, self.hidden_layers, dtype=torch.float32, device=device)
        ht1 = torch.zeros(batch_size, self.hidden_layers, dtype=torch.float32, device=device)
        ct1 = torch.zeros(batch_size, self.hidden_layers, dtype=torch.float32, device=device)

        for inpt in input_tensor.split(batch_size, dim=0):

            ht0, ct0 = self.lstm0(inpt, (ht0, ct0))
            ht1, ct1 = self.lstm1(ht0, (ht1, ct1))
            output = self.linear(ht1)
            predictions.append(output)

        for _ in range(future_preds):
            ht0, ct0 = self.lstm0(output, (ht0, ct0))
            ht1, ct1 = self.lstm1(ht0, (ht1, ct1))
            output = self.linear(ht1)

            predictions.append(output)

        predictions_tensor = torch.cat(predictions, dim=0).to(device)
        return predictions_tensor


def training(
    num_epochs, model, optimiser, loss_fn, train_input, train_target, val_input, val_target
):
    loss_out = []
    val_loss = []
    for epoch in range(num_epochs):
        model.train()
        optimiser.zero_grad()
        res = model(train_input, batch_size=5)
        train_loss = loss_fn(res, train_target)
        train_loss.backward()
        optimiser.step()
        loss = train_loss.item()
        loss_out.append(loss)

        model.eval()
        with torch.no_grad():
            vres = model(val_input[:-1], batch_size=125)
            vloss = loss_fn(vres, val_target[:-1]).item()
            val_loss.append(vloss)

        print(f"{epoch=} | {loss=} | {vloss=}")
    return (loss_out, val_loss)


def transform_to_lagged_seqs(arr, seq_len):
    xdim = len(arr) - seq_len

    x = np.zeros((xdim, seq_len))
    y = np.zeros((xdim, seq_len))

    for i in range(xdim):
        x[i, :] = arr[i : i + seq_len].flatten()
        y[i, :] = arr[i + 1 : i + seq_len + 1].flatten()

    xret = torch.from_numpy(x.reshape(-1, seq_len)).float().to(device)
    yret = torch.from_numpy(y.reshape(-1, seq_len)).float().to(device)

    return xret, yret


def add_to_database(name, train_loss, val_loss, actual, predicted, forecast):
    if Path("./database.json").is_file():
        df = pd.read_json("database.json")
        if name in df.columns:
            df.drop(columns=[name], inplace=True)
    else:
        df = pd.DataFrame()

    nf = pd.concat([train_loss, val_loss, actual, predicted, forecast], axis=1)
    nf["ticker"] = [name] * len(nf)
    gb = nf.groupby("ticker").agg(lambda x: x.dropna().to_dict())

    if not df.empty:
        stacked = pd.concat([df.T, gb], axis=0)

        stacked.T.to_json("database.json")
    else:
        gb.T.to_json("database.json")


def main(ticker, time_period):

    filename = f"{time_period}_{ticker}_stock_prices.csv"

    try:
        df = pd.read_csv(f"./data/{filename}")
    except FileNotFoundError:
        print("File not found. Please provide a valid file (csv) or use fetch_data.py.")
        sys.exit(1)

    close_prices = df.loc[:, "Close"].values

    trading_days_total = close_prices.shape[0]
    trading_days_per_year = 252

    ntrain = trading_days_total - 2 * trading_days_per_year

    nval = trading_days_per_year

    ntest = trading_days_per_year

    train_data = close_prices[:ntrain].reshape(-1, 1)
    val_data = close_prices[ntrain : ntrain + nval].reshape(-1, 1)
    test_data = close_prices[ntrain + nval :].reshape(-1, 1)

    scaler = MinMaxScaler()

    train_data = scaler.fit_transform(train_data)
    val_data = scaler.fit_transform(val_data)
    test_data = scaler.fit_transform(test_data)

    xtrain, ytrain = transform_to_lagged_seqs(train_data, 1)
    xval, yval = transform_to_lagged_seqs(val_data, 1)
    xtest, ytest = transform_to_lagged_seqs(test_data, 1)

    model = LSTM()

    model.to(device)

    lossfn = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=2e-2)

    num_epochs = 300
    nfuture = 20

    train_loss, val_loss = training(num_epochs, model, optim, lossfn, xtrain, ytrain, xval, yval)

    model.eval()

    with torch.no_grad():
        pred = model(xtest, future_preds=nfuture)
        nn_prediction = pred.detach().cpu().numpy()

    pred_rescaled = scaler.inverse_transform(nn_prediction)

    rmse = np.sqrt(
        mean_squared_error(close_prices[ntrain + nval : -1], pred_rescaled[:-nfuture].flatten())
    )

    print(f"\n RMSE = {rmse}")

    dloss = pd.DataFrame(train_loss, columns=["train_loss"])
    dvloss = pd.DataFrame(val_loss, columns=["val_loss"])
    dactual = pd.DataFrame(close_prices[ntrain + nval :], columns=["actual"])
    dpred = pd.DataFrame(pred_rescaled[:ntest].flatten(), columns=["predicted"])

    dfore = pd.DataFrame(
        pred_rescaled[ntest - 1 :].flatten().T,
        index=np.arange(ntest, ntest + nfuture),
        columns=["forecast"],
    )

    add_to_database(ticker, dloss, dvloss, dactual, dpred, dfore)

    plt.rc("font", family="serif", size=16)
    plt.rc("lines", linewidth=4, aa=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    title = f"${ticker} daily value prediction"

    fig.suptitle(title)

    ax[0].plot(np.arange(num_epochs), train_loss, label="Training")
    ax[0].plot(np.arange(num_epochs), val_loss, label="Validation")
    ax[0].set_ylabel("MSE Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend(title=f"RMSE={rmse:.3f}", title_fontsize=14, fontsize=14, frameon=False)

    ax[1].plot(close_prices[ntrain + nval :], label="actual")
    ax[1].plot(pred_rescaled[:ntest], label="predicted")
    ax[1].plot(
        np.arange(ntest, ntest + nfuture), pred_rescaled[ntest - 1 :], label="forecast"
    )
    ax[1].set_ylabel("Value (USD)")
    ax[1].set_xlabel("Days")
    ax[1].legend(fontsize=14, frameon=False)

    plt.tight_layout()
    plt.savefig(f"./plots/{ticker}_pred.png", dpi=600)
    plt.show()


if __name__ == "__main__":

    args = parser.parse_args()

    main(**vars(args))
