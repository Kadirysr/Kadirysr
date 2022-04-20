# !pip install mlxtend
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Görev 1
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()


# bir değişkenin eşik değerlerini belirliyor.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# belirlediği threshold değerlerini aykırı değerler ile değiştiriyor.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# veriyi ön işleme işlemleri
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


# NOT: yukarıdaki 3 fonksiyonu tekrar tekrar çağırmak yerine, Helpers klasör altına tanımlama yapılıp buradan çağrılabilir.
# from helpers.data_prep import outlier_thresholds

df = retail_data_prep(df)

############################################
# ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################


# Görev 2 - Germany müşterileri üzerinden birliktelik kuralları üretiniz.
df_ger = df[df['Country'] == "Germany"]
df_ger.head()

# invoice ve descriptionlara göre grupladık, aynı ürünler var ise sum ile topladık. Bir fişte bir üründen kaç tane var bulmuş olduk.
df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# iloc ilk 5 gözlemi görebilmek için kullanıyoruz.unstack veriyi sütunlara çekiyor.
df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

# NaN değerlerini 0 ile replace ediyoruz.
df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# Varsa 1 yoksa 0 yazdırmak için applymap kullanıyoruz.apply satır ya da sütunda gezmeyi sağlar.
# applymap bütün hücrelerde gezebilmek için kullanılıyor. SQL deki pivotlama işlemine karşılık gelir.
df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


# Bu işlemi fonksiyonlaştırırsak. ID false olursa Description'a göre işlem yapacak. Id girilmiş ise StockCode'a göre yapılacak.
# Daha rahat anlaşılması için ID yerine stockCode kullandım.
def create_invoice_product_df(dataframe, stockCode=False):
    if stockCode:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


ger_inv_pro_df = create_invoice_product_df(df_ger)
ger_inv_pro_df.head(5)

ger_inv_pro_df = create_invoice_product_df(df_ger, stockCode=True)
ger_inv_pro_df.head(5)

############################################
# Birliktelik Kurallarının Çıkarılması
############################################

# Apriori Algoritması- invoice product matriksi verdiğimizde bize support u hesaplar. min_support 0.01 altındakileri dışarıda bırakır.
# use_colnames - sütun isimlerini göster.
#support : frekans / fiş sayısı
frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()

# Yorum: 538 numaralı ürün tüm satın almaların % 81 inde varmış.

# association_rules - confidence, lift hesaplamalarını yapar. Support değerleri üzerinden hesaplar.
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
#Yorum - 22326 numaralı ürünleri alan müşteri POST ürününü alma olasılığı %91
#POST ürününü alan müşterinin 22326 numaralı ürünü alma olasılığı %27

# antecedents - antecedent support - tek başına gözükme olasılığı
# consequents - consequent support - tek başına gözükme olasılığı
# support - iki ürünün gözükme olasılığı
# confidence - 1. ürün alındığında 2. ürünün alınma olasılığı. lifti besler.
# lift - 1. ürün alındığında 2. ürünün alınma olasılığının kaç kat arttığını gösterir.
# leverage - kaldıraç - lifte benzer. supportu yüksek olan değerlere öncelik verme eğiliminde olması.
# conviction - 2. ürün olmadan 1. ürünün beklenen frekansı.
rules.sort_values("lift", ascending=False).head(500)


# Görev 3 ID'leri verilen ürünlerin isimleri nelerdir?
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_ger, 22328)
check_id(df_ger, 23235)
check_id(df_ger, 22747)

# Görev 4: Sepetteki kullanıcılar için ürün önerisi yapınız.
############################################
# Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################
product_id = 21987
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

sorted_rules.head()

recommendation_list = []

# Ürünleri (Antecedents için) listeye çevirerek içinde geziyor.
# Eğer yukarıda tanımlı product_id denk gelirse, ilgili satırdaki consequents içindeki ilk ürünü alır ve recommendation_list e ekler.
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
            #print(sorted_rules.iloc[i]["lift"], sorted_rules.iloc[i]["consequents"])

recommendation_list[0:1]

# Görev 5: Önerilen ürünlerin isimleri nelerdir?
check_id(df, "85049E")

check_id(df, recommendation_list[0])


# Görev 4 ve 5 in fonksiyon hali:
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]


check_id(df, 23049)
arl_recommender(rules, 21987, 1)
arl_recommender(rules, 23235, 2)
arl_recommender(rules, 22747, 3)

check_id(df, 22419)

# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747
