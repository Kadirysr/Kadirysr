import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#########################
# ODEVLER
#########################

###############################################
# ÖDEV 1: Fonksiyonlara Özellik Eklemek.
###############################################
df = pd.read_csv("datasets/titanic.csv")

def cat_summary(dataframe, col_name, plot=False):
    print("##################### Start #####################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title('test')
        plt.show()


#####  plotHeader Özelliği eklendi
def cat_summary2(dataframe, col_name, plot=False, plotHeader=False):
    print("##################### Start #####################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plotHeader:
        plt.title(col_name + ' Summary')

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


        cat_summary2(df, "Survived", plot=True,plotHeader=True)

###############################################
# ÖDEV 2: Docstring.
###############################################
# Aşağıdaki fonksiyona 4 bilgi (uygunsa) barındıran numpy tarzı docstring yazınız.
# (task, params, return, example)
# cat_summary()
def cat_summary(dataframe, col_name, plot=False):
    '''
      Returns the ratio and sum of specified column in dataset.

              Parameters:
                      dataframe: Dataframe
                            the dataframe which ratio and sum of column will be calculated
                      col_name: [Str]
                            the name of column to be calculated
                      plot:bool, optional
                            graph to be displayed (default is False)

              Returns:
                      dataframe: ratio and sum of specified column in dataframe

              Example:
                      cat_summary(df, "Survived", plot=True)

      '''
 print("##################### Start #####################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name],data = dataframe)
        plt.show()



##################PROJE KURAL TABANLI SINIFLANDIRMA###############################################

#Görev 1:
#Aşağıdaki soruları yanıtlayınız.
#Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("datasets/persona.csv")
df.head()
df.tail(5)
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

#Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

#Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()

#Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
print(df.groupby("PRICE").agg({"PRICE": "count"}))

#VEYA function tanımlayabiliriz
    def PriceSales(dataframe, target, numerical_col):
    print(pd.DataFrame({"SALES":dataframe.groupby(numerical_col)[target].count()}), end="\n\n\n" )

    PriceSales(df, "PRICE", "PRICE")

#Soru 5: Hangi ülkeden kaçar tane satış olmuş?
print(df.groupby("COUNTRY").agg({"PRICE": "count"}))

#VEYA üstteki function
PriceSales(df,"PRICE","COUNTRY")

#Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
print(df.groupby("COUNTRY").agg({"PRICE": "sum"}))

#Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
print(df.groupby("SOURCE").agg({"PRICE": "count"}))

#Soru 8: Ülkelere göre PRICE ortalamaları nedir?
print(df.groupby("COUNTRY").agg({"PRICE": "mean"}))

#Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
print(df.groupby("SOURCE").agg({"PRICE": "mean"}))

#Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
print(df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE": "mean"}))

#Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#virgülden sonra 2 basamak için round kullandım.
print(df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"}).round(2))

#Görev 3: Çıktıyı PRICE’a göre sıralayınız.
df1 = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"}).round(2)
agg_df = df1.sort_values ("PRICE",ascending=False)
agg_df

#Görev 4: Index’te yer alan isimleri değişken ismine çeviriniz.
agg_df = agg_df.reset_index()

#Görev 5: age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, agg_df["AGE"].max()],
                           labels=["0_18", "19_23", "24_30", "31_40", "41_" + str(agg_df["AGE"].max())])
agg_df.head()

#Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
agg_df["CUSTOMERS_LEVEL_BASED"]= agg_df.apply(lambda row:str(row['COUNTRY']).upper() +"_"+ str(row['SOURCE']).upper() +"_"+ str(row['SEX']).upper()+"_"+ str(row['AGE_CAT']).upper(), axis = 1)

agg_df = agg_df[["CUSTOMERS_LEVEL_BASED", "PRICE"]]

#tekilleştirme işlemi için mean alıyoruz:
agg_df = agg_df.groupby("CUSTOMERS_LEVEL_BASED").agg({"PRICE":"mean"})
agg_df = agg_df.reset_index()


#Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.

#Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız. Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D","C","B","A"])

# Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum"]})

#C segmentini analiz ediniz (Veri setinden sadece C segmentini çekip analiz ediniz).
agg_df_C = agg_df[agg_df["SEGMENT"]=="C"]
agg_df_C.groupby("SEGMENT").agg({"PRICE": ["mean","max","min","sum"]})

#Görev 7: Yeni gelen müşterileri segmentlerine göre sınıflandırınız ve ne kadar gelir getirebileceğini tahmin ediniz.

#▪ 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"]==new_user]

new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"]==new_user2]



