# Salary Prediction with Machine Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

df = pd.read_csv("../input/hitters/hitters.csv")
df.head()

# 1- DATA PROCESSING
df.shape

df.describe().T


# analysis of data
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# make upper all column names
df.columns = [col.upper() for col in df.columns]
df.head(2)

df["SALARY"].hist(bins=50)
plt.show()


##################################
# Grab numeric and categoric variables
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################################
# Correlation Analyssis
##################################
df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# Ranking of the variables with the highest correlation with the dependent variable salary: CRBI, CRuns, Chits, CAtBat


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)


##################################
# KATEGORİK DEĞİŞKENLERİN BAĞIMLI DEĞİŞKENE GÖRE ANALİZİ
##################################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({target + " Mean": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "SALARY", col)


# Result:
# A league salary mean > N league
# E position Salary > W position salary
# A legue Salary > N league Salary


##################################
# NÜMERİK DEĞİŞKENLERİN BAĞIMLI DEĞİŞKENE GÖRE ANALİZİ
##################################
# 1. way to analyze
# def target_summary_with_num(dataframe, target, numerical_col):
#    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
#
# for col in num_cols:
#    target_summary_with_num(df, "SALARY", col)

# 2. way to analyze

# This function gives the distribution of the numerical variables in the data set
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print("\t\t\t" + f"{numerical_col}" + " Nümerik Değişken Özet İstatistiği")
    print("\t\t\t---------------------------------------------")
    print(pd.DataFrame(dataframe[numerical_col].describe(quantiles)).T, "\n")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


# This function gives the distribution of the numerical variables and the relationship with variable
def num_analyser_plot(df, num_col, target_col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[num_col], kde=True, bins=30, ax=axes[0]);
    axes[0].lines[0].set_color('green')
    axes[0].set_title(f"{num_col}" + " " + "Dağılımı")
    axes[0].set_ylabel("Gözlem Sayısı")

    quantiles = [0, 0.25, 0.50, 0.75, 1]
    num_df = df.copy()
    num_df[f"{num_col}" + "_CAT"] = pd.qcut(df[num_col], q=quantiles)  # nümerik değişken kategorize
    df_2 = num_df.groupby(f"{num_col}" + "_CAT")[target_col].mean()

    sns.barplot(x=df_2.index, y=df_2.values);
    axes[1].set_title(f"{num_col} Kırılımında {target_col} Ortalaması")
    axes[1].set_ylabel(f"{target_col}")
    plt.show()


num_summary(df, "HITS")
num_analyser_plot(df, "HITS", "SALARY")

num_summary(df, "RBI")
num_analyser_plot(df, "RBI", "SALARY")

num_summary(df, "CRUNS")
num_analyser_plot(df, "CRUNS", "SALARY")


# It is a graph function that shows outliers in numerical variables in the data set.
# Veri setindeki nümerik değişkenlerdeki aykırı gözlemleri gösteren grafik fonksiyonudur.
def outliers_boxplot(dataframe, num_cols):
    plt.figure(figsize=(12, 6), dpi=200)
    plt.title("Outliers Analysis for numeric variables")
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=df.loc[:, num_cols], orient="h", palette="Set3")
    plt.show()


df_mean = int(df.mean(numeric_only=True).quantile(0.5))
high_cols = []
low_cols = []
for col in num_cols:
    if df[col].mean() >= df_mean:
        high_cols.append(col)
    else:
        low_cols.append(col)

# Yüksek değerlikli nümerik değişkenlerin aykırı gözlem analizi
outliers_boxplot(df, high_cols)

# Düşük değerlikli nümerik değişkenlerin aykırı gözlem analizi
outliers_boxplot(df, low_cols)


##################################
# Aykırı Gözlem Analizi
##################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# correlation
corr_df = df.corr()
corr_df["SALARY"].sort_values(ascending=False)
corr_df.head()

##################################
# ENCODING
##################################

# ONE HOT ENCODING
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
ohe_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)


# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# RARE ENCODING
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "SALARY", cat_cols)

# Rare olarak kabul edilebilecek (ratio < 0.01 seviyede) herhangi bir değişken gözlemi gözükmüyor.


##################################
# 2-Feature Engineering
##################################
# Yukarıda veri analiz edildi, sadece aykırı değerler düzeltildi.
# Bundan sonra anlamlı değişkenler oluşturulacak ve model için değişkenler sayısal hale getirilecektir.

##################################
# Yeni Değişkenler Üretme
##################################a
# Bu kısımda yeni değişkenler üretilecek. Bu kısımda korelasyonu yüksek değişkenlerle yeni değişken üretmek daha faydalı olacaktır.

# Kariyer istatistiklerinden 86-87 sezonu istatistiklerini çıkartıyoruz.
df["NEW_CATBAT"] = df["CATBAT"] - df["ATBAT"]
df["NEW_CHITS"] = df["CHITS"] - df["HITS"]
df["NEW_CHMRUN"] = df["CHMRUN"] - df["HMRUN"]
df["NEW_CRUNS"] = df["CRUNS"] - df["RUNS"]
df["NEW_CRBI"] = df["CRBI"] - df["RBI"]
df["NEW_CWALKS"] = df["CWALKS"] - df["WALKS"]

# bölme işlemlerinde sorun yaşamamak için 1 ekliyoruz.
df = df + 1

new_cols = [col for col in df.columns if col.startswith("C") == False]

df = df[new_cols]

df.head(5)

df = df[new_cols]

corr_df = df.corr()
corr_df["SALARY"].sort_values(ascending=False)

df['NEW_ATBAT_HITS_RUNS'] = (df['ATBAT'] * corr_df.loc[corr_df.index=="ATBAT", "SALARY"].values+ df['HITS'] * corr_df.loc[corr_df.index=="HITS", "SALARY"].values
                             + df['RUNS'] * corr_df.loc[corr_df.index=="RUNS", "SALARY"].values)/(corr_df.loc[corr_df.index=="ATBAT", "SALARY"].values
                                                                                                  + corr_df.loc[corr_df.index=="HITS", "SALARY"].values + corr_df.loc[corr_df.index=="RUNS", "SALARY"].values)

df['NEW2_CHIT_RATE']=df["NEW_CHITS"]/df["YEARS"]
df['NEW2_CHMRUN_RATE']=df["NEW_CHMRUN"]/df["YEARS"]
df['NEW2_CRUNS_RATE']=df["NEW_CRUNS"]/df["YEARS"]
df['NEW2_CRBI_RATE']=df["NEW_CRBI"]/df["YEARS"]
df['NEW2_CWALKS_RATE']=df["NEW_CWALKS"]/df["YEARS"]


df['NEW_HIT_RATE']=df["NEW_CHITS"]/df["NEW_CATBAT"]
df['NEW_RBI_RATE']=df["RBI"]/df["NEW_CRBI"]
df['NEW_WALKS_RATE']=df["WALKS"]/df["NEW_CWALKS"]
df['NEW_HITS_RATE']=df["HITS"]/df["NEW_CHITS"]
df['NEW_HMRUN_RATE']=df["HMRUN"]/df["NEW_CHMRUN"]
df['NEW_RUNS_RATE']=df["RUNS"]/df["NEW_CRUNS"]
df['NEW_CHMRUN_RATE']=df["NEW_CHMRUN"]/df["NEW_CHITS"]
df['NEW_Cat_RATE']=df["NEW_CRUNS"]/df["NEW_CATBAT"]

df['NEW_ASSISTS/ERRORS']= df["ASSISTS"]/df["ERRORS"]
df['NEW_CRBI/CHITS']= df["NEW_CRBI"]/df["NEW_CHITS"]
df['NEW_CRUNS/CHITS']= df["NEW_CRUNS"]/df["NEW_CHITS"]

df['NEW_HITRATIO'] = df['HITS'] / df['ATBAT']


df['NEW_CRUNRATIO'] =  df['NEW_CRUNS'] /df['NEW_CHMRUN']

df['NEW_AVG_ATBAT'] = df['NEW_CATBAT']/df['YEARS']
df['NEW_AVG_HITS'] = df['NEW_CHITS']/df['YEARS']
df['NEW_AVG_HMRUN'] = df['NEW_CHMRUN']/df['YEARS']
df['NEW_AVG_RUNS'] = df['NEW_CRUNS'] / df['YEARS']
df['NEW_AVG_RBI'] = df['NEW_CRBI'] / df['YEARS']
df['NEW_AVG_WALKS'] = df['NEW_CWALKS'] / df['YEARS']


df['NEW_Perf_HITS'] = (df['HITS'] - df['NEW_AVG_HITS']) / df['HITS']
df['NEW_Perf_RUNS'] = (df['RUNS'] - df['NEW_AVG_RUNS']) / df['RUNS']
df['NEW_Perf_RBI'] = (df['RBI'] - df['NEW_AVG_RBI']) / df['RBI']
df['NEW_Perf_WALKS'] = (df['WALKS'] - df['NEW_AVG_WALKS']) / df['WALKS']

df.head(5)

# missing value var mı?
df.isnull().values.any()

#  veri seti boyutu
df.shape

#checking for null values
df.isnull().sum()

#dropping null values
df.dropna(inplace=True)

# missing value'ların atılması sonrası veri seti boyutu
df.shape

##################################
# Standartlaştırma
##################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove("SALARY")
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head(3) # Encoding sonrası veri seti

##################################
# MODEL OLUŞTURMA
##################################

#Lineer Regresyon Modeli Kurma
y = df["SALARY"]  # Bağımlı değişken
X = df.drop(["SALARY"], axis=1)  # Bağımsız değişkenler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

y_train.mean(), y_train.median(), y_train.std(), y_test.mean(), y_test.median(), y_test.std()

y_train.hist(bins=50)
plt.show()

y_test.hist(bins=20)
plt.show()

#Model Kurulumu
lin_model = LinearRegression().fit(X_train, y_train)

# b0 sabit sayısı
print('intercept:', lin_model.intercept_)

# Denklem katsayıları
print('coefficients:', lin_model.coef_)

#Model Denklemi
print("Model Denklemi:\t"+
f"Y= {lin_model.intercept_:.3f}{lin_model.coef_[0]:.3f}X{lin_model.coef_[0]:.3f}X+------+{lin_model.coef_[df.shape[1]-2]:.3f}X")


coefs = pd.DataFrame( lin_model.coef_, columns=['Coefficients'], index=X_train.columns)
coefs.plot(kind='barh', figsize=(18, 12))
plt.title('Lasso model, strong regularization')
plt.axvline(x=0, color='.9')
plt.subplots_adjust(left=.3)

#Yüksek ağırlıklı ilk 5 değişken
coefs.sort_values("Coefficients", ascending=False).head(5)

#Prediction Test
random_user = X.sample(1, random_state=11)  # Random player from dataset
lin_model.predict(random_user)  # Prediction of Salary

view_df = pd.read_csv("../input/hitters/hitters.csv")
view_df.loc[random_user.index]

##################################
# SUCCESS OF PREDICTION
##################################

# R-Kare Result on Train Data Set
lin_model.score(X_train, y_train)

# RMSE Result on Train Data Set
y_pred = lin_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# R-Kare Result on TEST Data Set
lin_model.score(X_test, y_test)

#  RMSE Result on TEST Data Set
y_pred = lin_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

np.mean(np.sqrt(-cross_val_score(lin_model, X, y, cv=10, scoring="neg_mean_squared_error")))  #  CV RMSE Result

#As a result, an average of 72% success and 243 RMSE error measurement values.