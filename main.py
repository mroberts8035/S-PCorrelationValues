import datetime as dt
import time
from datetime import date
from datetime import timedelta
import os
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import pandas as pd
from pandas_datareader import data as pdr
import bs4 as bs
import pickle
import requests
import yfinance as yf
yf.pdr_override()
style.use('ggplot')


def save_sp500_tickers():
    response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(response.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.strip())

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    # print(tickers)

    return tickers


save_sp500_tickers()


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
        #if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
        df = pdr.get_data_yahoo(ticker, start, end)
        df.to_csv('stock_dfs/{}.csv'.format(ticker))
        time.sleep(.005)
        #else:
            #print('Already have {}'.format(ticker))


get_data_from_yahoo()
