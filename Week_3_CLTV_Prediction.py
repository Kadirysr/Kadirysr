
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
#from sklearn.preprocessing import MinMaxScaler



def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# credentials.
creds = {'user': 'group_8',
         'passwd': 'miuul',
         'host': '34.79.73.237',
         'port': 3306,
         'db': 'group_8'}


connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

conn = create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;", conn)


retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)

retail_mysql_df.head()
retail_mysql_df.info()
df = retail_mysql_df.copy()
df.head()

df = df[df["Country"] == "United Kingdom"]
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C",na=False)]
df = df[df["Quantity"]>0]
df = df[df["Price"]>0]

replace_with_thresholds(df,"Quantity")
replace_with_thresholds(df,"Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7



bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


#6  Aylık CLTV prediction
cltv_df['expected_purc_1_week'] = bgf.conditional_expected_number_of_purchases_up_to_time(24,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)




#1 aylık ve 12 aylık - ilk 10 kişiler
cltv_df['expected_purc_1_month'] =bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df['expected_purc_12_month'] =bgf.conditional_expected_number_of_purchases_up_to_time(48,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df.head()

#Gamma Gamma Modeli !!!!
ggf =GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'],cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

#CLTV Hesaplanması
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=month,
                                   freq = "W",
                                   discount_rate=0.01)
cltv.head()
cltv.shape
cltv = cltv.reset_index()
cltv.head()

cltv_final = cltv_df.merge(cltv, on="customerID", how="left")
#cltv_final.sort_values(by="clv",ascending=False)

#cltv ye göre segmentlere ayır
cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"]

#Gönder veri tabanına

cltv_final.to_sql(name='kadir_yasar', con=conn, if_exists='replace', index=False)

