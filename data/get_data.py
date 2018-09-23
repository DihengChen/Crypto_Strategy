# -*- coding: utf-8 -*-
# @Author: Sky Zhang
# @Date:   2018-09-22 22:16:19
# @Last Modified by:   Sky Zhang
# @Last Modified time: 2018-09-23 10:05:06

import pandas as pd
from requests import get
import re
import json
import datetime
import time
import pickle


def get_coinlist():
    url = 'https://min-api.cryptocompare.com/data/all/coinlist'
    page = get(url).text
    x = re.findall("\{\"Id\".*?\}", page)
    table = {"CoinName": [], "Symbol": [], "IsTrading": []}
    for i in range(len(x)):
        cache = json.loads(x[i])
        for key in table:
            table[key].append(cache[key])
    table = pd.DataFrame(table)
    return table


def daily_price_historical(end_time, symbol, comparison_symbol="USD",
                           limit=2000, aggregate=1, allData='true'):
    timestamp = time.mktime(datetime.datetime.strptime(end_time,
                                                       "%Y-%m-%d").timetuple())
    url = ('https://min-api.cryptocompare.com/data/histoday?' +
           'fsym={}&tsym={}&limit={}&aggregate={}&allData={}&toTs={}')
    url = url.format(symbol.upper(), comparison_symbol.upper(),
                     limit, aggregate, allData, timestamp)
    page = get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['time'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    df = df.loc[:, ["open", "high", "low", "close", "time"]]
    df.time = df.time.dt.strftime('%Y-%m-%d')
    df.set_index("time", inplace=True)
    return df


def main():
    coinlist = get_coinlist()
    coinlist.to_pickle("coinlist.pkl")
    end_time = "2018-09-20"

    database = {}
    for i in range(len(coinlist)):
        if (i + 1) % 1000 == 0:
            time.sleep(10 * 60)
            print("wait for 10 min...")
        if coinlist.iloc[i, 1]:
            # print("downloading {}'s data ({}/{})".format
            # (coinlist.iloc[i,2],i+1,n))
            try:
                a = coinlist.iloc[i, 2]
                database[a] = daily_price_historical(end_time,
                                                     coinlist.iloc[i, 2])
            except Exception as e:
                print(e)
                print("continue downloading... \n")
            finally:
                time.sleep(1.5)

    with open("hist_daily_ohlc.pkl", "wb") as f:
        pickle.dump(database, f)


if __name__ == "__main__":
    main()
