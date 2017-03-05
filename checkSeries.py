import pandas as pd
import datetime
import numpy as np
pd.set_option('display.max_columns', None)

def main():
    dateparse = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')

    df = pd.read_csv('all_data-3-1.txt', parse_dates=0, index_col=0,date_parser=dateparse, delimiter=',')

    index = pd.date_range('2016-08-15 07:00:00', '2017-03-01 09:45:00', freq='15min')

    df.index = pd.DatetimeIndex(df.index)
    
    
    df = df.reindex(index, fill_value=np.nan)

    nan_rows = df[df.isnull().T.any().T]
    print nan_rows

if __name__ == '__main__':
    main()
