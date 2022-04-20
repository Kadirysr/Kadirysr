import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv",
                 low_memory=False)  # DtypeWarning kapamak icin
df.head()

df["overall"].value_counts()

# Direk ortalama alıyoruz. Bu sağlıklı bir değerlendirme değil.
df.groupby("asin").agg({"asin": "count",
                        "overall": "mean"})

# tek ürün olduğu için aşağıdaki gibi alabiliriz.
df["overall"].mean()

# Time Based ortalama hesaplama yapıyoruz. En taze yorum en fazla puanı alır.

# info ile baktığımızda reviewTime veri tipini object olarak geliyor. önce bunu datetime a çevirelim.
df.info()

df['reviewTime'] = pd.to_datetime(df['reviewTime'])

# dataset içindeki max zamanı buluyorum.+1 ekleyerek current time belirliyorum.
df.groupby("asin").agg({"reviewTime": "count",
                        "reviewTime": "max"})

current_date = pd.to_datetime('2014-12-08 0:0:0')

# days column oluşturuyoruz. yorumların ne kadar zaman önce yazıldığı hesaplanıyor.
df["days"] = (current_date - df['reviewTime']).dt.days

# days verisine bir göz atalım.
df.sort_values("days").head(20)

Q1 = df["days"].quantile(0.25)
Q2 = df["days"].quantile(0.5)
Q3 = df["days"].quantile(0.75)


# time based olarak hesaplama yaptırıyoruz ortalamayı.
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= Q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > Q2) & (dataframe["days"] <= Q3), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > Q3), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

df["overall"].mean()

# ORTALAMA YORUM: review zamanlarına göre ortalama hesabı yapıldığında ortalama 3.48, genel ortalamaya bakıldığında ise 4.58 görülüyor. Demekki son zamanlara doğru daha düşük puan almış.

df.head()

# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
# önce total vote a göre sıralama yapıp ne kadar vote alınmış durum nedir bakalım. veriyi görelim.
df.sort_values("total_vote", ascending=False).head(20)


def score_avg(up, down):
    return up + down

score_avg(df["helpful_yes"], df["total_vote"]) .head(20)

df["score_avg"] = df["helpful_yes"] / df["total_vote"]

df.head(20)