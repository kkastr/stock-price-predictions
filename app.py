import streamlit as st
import pandas as pd
import numpy as np
import glob
from utils import getData, fmt_path_strings, DateShortName
from nn_pred import TrainModel
st.title('Stock Prediction Dashboard')

den = st.container()

den_c1, den_c2, den_c3 = den.columns(3)

ticker = den_c1.text_input('Add a new ticker')
start_date = den_c2.date_input("Start date")
end_date = den_c3.date_input("End date")

dl_btn = den.button("Download Data")

if dl_btn:
    args = dict(ticker=ticker, start_date=start_date, end_date=end_date, update=False)
    getData(**args)


dsel = st.container()

dsel_c1, dsel_c2, dsel_c3 = dsel.columns(3)

curr_data = fmt_path_strings(glob.glob("./data/*.csv"))


stock_select = dsel_c1.selectbox("Pick a stock", curr_data, label_visibility="collapsed")
train_btn = dsel_c2.button("Train Model")
# plot_btn = dsel_c3.button("Plot Results")


if train_btn:

    args = dict(ticker=ticker, time_period='5Y', vis=False)
    TrainModel(**args)
