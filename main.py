import datetime as dt
import pathlib
import time
import seaborn as sns
from datetime import date
from datetime import timedelta
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
from pandas_datareader import data as pdr
import bs4 as bs
import pickle
import requests
import yfinance as yf
yf.pdr_override()
style.use('ggplot')


# The function below goes to Wikipedia and scrapes all the ticker symbols for the S&P500 index, saves them into a
# .pickle file to be used later in the app.
# Uses the libraries: BS4, requests, and pickle
def save_sp500_tickers():
    response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(response.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.replace('.', '-').strip())

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    # print(tickers)

    return tickers


# Uncomment out the below function call if you need to update the tickers for the S&P 500
# save_sp500_tickers()


# The function below takes the tickers saved from save_sp500_tickers and for each one will scrape the stock data from
# Yahoo finance for the last 2 years and saves them in their own .csv file. This way we can access the data for other
# calculations or anything else we want to do to analyze the data at a later time without having to scrape the data all
# over again.
# Uses the libraries: yfinance, os, datetime, pandas_datareader, pickle
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = date.today() - timedelta(days=730)
    end = date.today() - timedelta(days=1)

    for ticker in tickers:
        df = pdr.get_data_yahoo(ticker, start, end)
        df.to_csv('stock_dfs/{}.csv'.format(ticker))
        time.sleep(.005)


# Uncomment the below function and run weekly to update the data when running correlation calculations.
# get_data_from_yahoo()


# The below function combines the Adjusted Close column from the CSV files for the tickers into one data file
#
# Using the libraries: pd,
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename({'Adj Close': ticker}, axis=1, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


# compile_data()


# For proof that we are able to access the data. Plotting a line chart of AAPL from the compiled data form.
# Then that proof is commented out and we produce a correlation chart using df.corr() and save that as data
# Also we produce a heatmap using red, yellow, green, color spectrum
# Uses matplotlib.pyplot, pd, and from matplotlib style, numpy
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    # df['AAPL'].plot()
    # plt.show()
    df_corr = df.corr()

    # print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

    # sns keeps utilizing too many resources
    # sns.heatmap(df_corr, annot=True, cmap='RdYlGn')
    # plt.show()


visualize_data()
