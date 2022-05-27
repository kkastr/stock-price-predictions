import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description='''Train NN to predict oruce and plot the result''')

parser.add_argument('-ticker', dest='ticker', type=str,
                    help='The ticker for the security (i.e. GOOG, AMZN, etc).')
parser.add_argument('-period', dest='time_period', type=str,
                    help='The period of time for the downloaded data (e.g. 5Y)')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers

        self.lstm1 = nn.LSTMCell(1, self.hidden_layers, device=device)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers, device=device)
        self.linear = nn.Linear(self.hidden_layers, 1, device=device)

    def forward(self, input_vals, batch_size=1, future_preds=0):
        predictions = []
        h_t = torch.zeros(batch_size, self.hidden_layers, dtype=torch.float32, device=device)
        c_t = torch.zeros(batch_size, self.hidden_layers, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(batch_size, self.hidden_layers, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(batch_size, self.hidden_layers, dtype=torch.float32, device=device)

        for input_t in input_vals.split(batch_size, dim=0):

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            predictions.append(output)

        for i in range(future_preds):
            # use the last input to predict future values
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            predictions.append(output)

        predictions_tensor = torch.cat(predictions, dim=0).to(device)
        return predictions_tensor


def training(n_epochs, model, optimiser, loss_fn, train_input):
    loss_out = []
    for i in range(n_epochs):
        optimiser.zero_grad()
        out = model(train_input, batch_size=16)
        loss = loss_fn(out, train_input)
        loss.backward()
        optimiser.step()

        loss_out.append(float(loss))
        print("Step: {}, Loss: {}".format(i, loss))
    return loss_out


def main(ticker, time_period):

    filename = f'{time_period}_{ticker}_stock_prices_.csv'

    df = pd.read_csv(f'./data/{filename}')

    close_prices = df.loc[:, 'Close'].values

    ntrain = 992
    ntest = close_prices.shape[0] - ntrain

    train_data = close_prices[:ntrain].reshape(-1, 1)
    test_data = close_prices[ntrain:].reshape(-1, 1)

    scaler = MinMaxScaler()

    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    train_input = torch.from_numpy(train_data).float().to(device)

    test_input = torch.from_numpy(test_data).float().to(device)

    model = LSTM()

    model.to(device)

    lossfn = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=2e-2)

    num_epochs = 150

    loss = training(num_epochs, model, optim, lossfn, train_input)

    nfuture = 14

    model.eval()

    with torch.no_grad():
        pred = model(test_input, future_preds=nfuture)
        nn_prediction = pred.detach().cpu().numpy()

    pred_rescaled = scaler.inverse_transform(nn_prediction)

    plt.rc('font', family='serif', size=20)
    plt.rc('lines', linewidth=4, aa=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    title = f'${ticker} daily value prediction'

    fig.suptitle(title)

    ax[0].plot(np.arange(num_epochs), loss)
    ax[0].set_ylabel('MSE Loss')
    ax[0].set_xlabel('Epoch')

    ax[1].plot(np.arange(ntest), close_prices[ntrain:], label='real')
    ax[1].plot(np.arange(ntest), pred_rescaled[:ntest], label='predicted')
    ax[1].plot(np.arange(ntest, ntest + nfuture), pred_rescaled[ntest:], label='future prediction')
    ax[1].set_ylabel('Value (USD)')
    ax[1].set_xlabel('Days')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots/{ticker}_pred.png', dpi=600)


if __name__ == "__main__":

    args = parser.parse_args()

    main(**vars(args))
