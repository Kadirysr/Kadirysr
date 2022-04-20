
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

##################################
# 2-Feature Engineering
##################################
# Yukarıda veri analiz edildi, sadece aykırı değerler düzeltildi.
# Bundan sonra anlamlı değişkenler oluşturulacak ve model için değişkenler sayısal hale getirilecektir.
​
##################################
# Yeni Değişkenler Üretme
##################################a
# Bu kısımda yeni değişkenler üretilecek. Bu kısımda korelasyonu yüksek değişkenlerle yeni değişken üretmek daha faydalı olacaktır.
​
# Kariyer istatistiklerinden 86-87 sezonu istatistiklerini çıkartıyoruz.
df["NEW_CATBAT"] = df["CATBAT"] - df["ATBAT"]
df["NEW_CHITS"] = df["CHITS"] - df["HITS"]
df["NEW_CHMRUN"] = df["CHMRUN"] - df["HMRUN"]
df["NEW_CRUNS"] = df["CRUNS"] - df["RUNS"]
df["NEW_CRBI"] = df["CRBI"] - df["RBI"]
df["NEW_CWALKS"] = df["CWALKS"] - df["WALKS"]
​
# bölme işlemlerinde sorun yaşamamak için 1 ekliyoruz.
df = df + 1
​
new_cols = [col for col in df.columns if col.startswith("C") == False]
​
df = df[new_cols]
​
df.head(5)
​
df = df[new_cols]
​
corr_df = df.corr()
corr_df["SALARY"].sort_values(ascending=False)
​
df['NEW_ATBAT_HITS_RUNS'] = (df['ATBAT'] * corr_df.loc[corr_df.index=="ATBAT", "SALARY"].values+ df['HITS'] * corr_df.loc[corr_df.index=="HITS", "SALARY"].values
                             + df['RUNS'] * corr_df.loc[corr_df.index=="RUNS", "SALARY"].values)/(corr_df.loc[corr_df.index=="ATBAT", "SALARY"].values
                                                                                                  + corr_df.loc[corr_df.index=="HITS", "SALARY"].values + corr_df.loc[corr_df.index=="RUNS", "SALARY"].values)
​
df['NEW2_CHIT_RATE']=df["NEW_CHITS"]/df["YEARS"]
df['NEW2_CHMRUN_RATE']=df["NEW_CHMRUN"]/df["YEARS"]
df['NEW2_CRUNS_RATE']=df["NEW_CRUNS"]/df["YEARS"]
df['NEW2_CRBI_RATE']=df["NEW_CRBI"]/df["YEARS"]
df['NEW2_CWALKS_RATE']=df["NEW_CWALKS"]/df["YEARS"]
​
​
df['NEW_HIT_RATE']=df["NEW_CHITS"]/df["NEW_CATBAT"]
df['NEW_RBI_RATE']=df["RBI"]/df["NEW_CRBI"]
df['NEW_WALKS_RATE']=df["WALKS"]/df["NEW_CWALKS"]
df['NEW_HITS_RATE']=df["HITS"]/df["NEW_CHITS"]
df['NEW_HMRUN_RATE']=df["HMRUN"]/df["NEW_CHMRUN"]
df['NEW_RUNS_RATE']=df["RUNS"]/df["NEW_CRUNS"]
df['NEW_CHMRUN_RATE']=df["NEW_CHMRUN"]/df["NEW_CHITS"]
df['NEW_Cat_RATE']=df["NEW_CRUNS"]/df["NEW_CATBAT"]
​
df['NEW_ASSISTS/ERRORS']= df["ASSISTS"]/df["ERRORS"]
df['NEW_CRBI/CHITS']= df["NEW_CRBI"]/df["NEW_CHITS"]
df['NEW_CRUNS/CHITS']= df["NEW_CRUNS"]/df["NEW_CHITS"]
​
df['NEW_HITRATIO'] = df['HITS'] / df['ATBAT']
​
​
df['NEW_CRUNRATIO'] =  df['NEW_CRUNS'] /df['NEW_CHMRUN']
​
df['NEW_AVG_ATBAT'] = df['NEW_CATBAT']/df['YEARS']
df['NEW_AVG_HITS'] = df['NEW_CHITS']/df['YEARS']
df['NEW_AVG_HMRUN'] = df['NEW_CHMRUN']/df['YEARS']
df['NEW_AVG_RUNS'] = df['NEW_CRUNS'] / df['YEARS']
df['NEW_AVG_RBI'] = df['NEW_CRBI'] / df['YEARS']
df['NEW_AVG_WALKS'] = df['NEW_CWALKS'] / df['YEARS']
​
​
df['NEW_Perf_HITS'] = (df['HITS'] - df['NEW_AVG_HITS']) / df['HITS']
df['NEW_Perf_RUNS'] = (df['RUNS'] - df['NEW_AVG_RUNS']) / df['RUNS']
df['NEW_Perf_RBI'] = (df['RBI'] - df['NEW_AVG_RBI']) / df['RBI']
df['NEW_Perf_WALKS'] = (df['WALKS'] - df['NEW_AVG_WALKS']) / df['WALKS']
​
df.head(5)
​
# missing value var mı?
df.isnull().values.any()
​
#  veri seti boyutu
df.shape
​
#checking for null values
df.isnull().sum()
​
#dropping null values
df.dropna(inplace=True)
​
# missing value'ların atılması sonrası veri seti boyutu
df.shape
​
##################################
# Standartlaştırma
##################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove("SALARY")
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head(3) # Encoding sonrası veri seti
​
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import precision_score,recall_score,accuracy_score


#####################################################
# Base Models
######################################################
y = df["SALARY"]  # Bağımlı değişken
X = df.drop(["SALARY"], axis=1)  # Bağımsız değişkenler
​
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
​
df_perf_metrics = pd.DataFrame(columns=[
    'Model', 'MSE'
])
models_trained_list = []
​
def get_perf_metrics(model, i):
    # model name
    model_name = type(model).__name__
    print("Training {} model...".format(model_name))
    # Fitting of model
    model.fit(X_train, y_train)
    print("Completed {} model training.".format(model_name))
    # Predictions
    y_pred = model.predict(X_test)
    # Add to ith row of dataframe - metrics
​
    df_perf_metrics.loc[i] = [
        model_name,
        np.sqrt(mean_squared_error(y_test, y_pred))
    ]
​
    print("Completed {} model's performance assessment.".format(model_name))
​
models_list = [LinearRegression(),Ridge(), Lasso(), ElasticNet(),  KNeighborsRegressor(), DecisionTreeRegressor(),
          RandomForestRegressor(),SVR(), GradientBoostingRegressor(), XGBRegressor(objective='reg:squarederror'), LGBMRegressor()]
​
​
for n, model in enumerate(models_list):
    get_perf_metrics(model, n)
	
df_perf_metrics


#Model	MSE
#0	LinearRegression	220.075
#1	Ridge	208.672
#2	Lasso	213.080
#3	ElasticNet	203.169
#4	KNeighborsRegressor	231.777
#5	DecisionTreeRegressor	319.018
#6	RandomForestRegressor	215.439
#7	SVR	399.295
#8	GradientBoostingRegressor	215.838
#9	XGBRegressor	247.155
#10	LGBMRegressor	219.225

######################################################
# Hyperparameter Optimization for Random Forests
######################################################
rf_model = RandomForestRegressor(random_state=42)

rf_params = {"max_depth": [5,10, 15, 20],
             "max_features": [3,5, 7,8, 15,"auto"],
             "min_samples_split": [15, 20,25,30],
             "n_estimators": [100,150,200,300]}

rf_cv = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)

rf_cv.best_params_

#{'max_depth': 5,
# 'max_features': 7,
# 'min_samples_split': 20,
# 'n_estimators': 150}

final_model = RandomForestRegressor(random_state=42, **rf_cv.best_params_).fit(X_train, y_train)
y_final_pred = final_model.predict(X_test)
rf_final_model_error = np.sqrt(mean_squared_error(y_test, y_final_pred))
rf_final_model_error

#200.89458319278398


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
plot_importance(final_model, X_train)

#1 NEW2_CHIT_RATE
#2 NEW_AVG_HITS
#3 NEW_AVG_RBI
#4 NEW2_CRUNS_RATE
#5 NEW_CRUNS












