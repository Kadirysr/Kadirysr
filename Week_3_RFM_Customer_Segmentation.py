import datetime as dt
import  pandas as pd


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.5f' % x)

#1. Online Retail II excelindeki 2010-2011 verisini okuyunuz. Oluşturduğunuz dataframe’in kopyasını oluşturunuz.
df_ = pd.read_excel("datasets/online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()

#2. Veri setinin betimsel istatistiklerini inceleyiniz.
df.head()
df.shape
df.describe().T
df["Invoice"].nunique()

#3. Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?
df.isnull().sum()

#4. Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.
df.dropna(inplace=True)

#5. Eşsiz ürün sayısı kaçtır?
df["Description"].nunique()

#6. Hangi üründen kaçar tane vardır?
df["Description"].value_counts().head()

#7. En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız.
 df.groupby("Description").agg({"Quantity": "sum"}).sort_values ("Quantity",ascending=False).head(5)


#8. Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.
df = df[~df["Invoice"].str.contains("C",na=False)]


#9. Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz.
df["TotalPrice"] = df["Price"] * df["Quantity"]


#Görev 2: RFM metriklerinin hesaplanması
df["InvoiceDate"].max()

today_date = dt.datetime(2010, 12, 11)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']

#Görev 3: RFM skorlarının oluşturulması ve tek bir değişkene çevrilmesi

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# 1 1,1,2,3,3,3,3,3,

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.head()

rfm.describe().T

rfm[rfm["RFM_SCORE"] == "55"].head()

rfm[rfm["RFM_SCORE"] == "11"].head()


#Görev 4: RFM skorlarının segment olarak tanımlanması

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm.head()

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "need_attention"].head()

rfm[rfm["segment"] == "new_customers"].index

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "loyal_customers"].index

new_df.to_csv("new_customers.csv")


