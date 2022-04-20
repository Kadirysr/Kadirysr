# Titanic Survival Model

# CONTENT:
# 1. Data Processing
#        1.1 Reading Data
#        1.2 Data Analysis
# 2. Feature Engineering & Data Pre-Processing
#        2.1 Feature Engineering
#        2.2 Outliers
#        2.3 Missing Values
#        2.4 Label Encoding
#        2.5 One-Hot Encoding
#        2.6 Rare Encoding
#        2.7 Standart Scaler
# 3. Logistic Regression
#        3.1 Model
#        3.2 Prediction
#        3.3 Success Evaluation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, \
    precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

######################################################
#  1. Data Processing
######################################################

######################################################
#  1.1 Reading Data
######################################################

df = pd.read_csv("../input/titanic/titanic.csv")
df.head()

######################################################
#  1.2 Data Analysis
######################################################
df.describe().T


# Data Analysis
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

# Data Analysis
# Make upper all column names
df.columns = [col.upper() for col in df.columns]
df.head(2)


##################################
# Grab numeric and categoric variables
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
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
num_cols

######################################################
# 2. Feature Engineering & Data Pre-Processing
######################################################

######################################################
# 2.1 Feature Engineering
######################################################

df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df['NEW_TITLE'] = df.NAME.str.extract('([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

df.head()


######################################################
# 2.2 Outliers
######################################################
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


num_cols = [col for col in num_cols if "PASSENGERID" not in col]

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

######################################################
# 2.3 Missing Values
######################################################

df.isnull().sum()  # control

missing_cols = [col for col in df.columns if (df[col].isnull().any()) & (col != "Cabin")]
for i in missing_cols:
    if i == "AGE":
        df[i].fillna(df.groupby("PCLASS")[i].transform("median"), inplace=True)
    elif df[i].dtype == "O":
        df[i].fillna(df[i].mode()[0], inplace=True)
    else:
        df[i].fillna(df[i].median(), inplace=True)

deleted_cols = ["CABIN", "SIBSP", "PARCH", "TICKET", "NAME"]
df = df.drop(deleted_cols, axis=1)

df.isnull().sum()  # control

df["NEW_AGE_CAT"] = pd.cut(df["AGE"], bins=[0, 20, 35, 55, df["AGE"].max() + 1],
                           labels=[1, 2, 3, 4]).astype(int)
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 20), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 20) & (df['AGE']) < 55), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 55), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 20), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 20) & (df['AGE']) < 55), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 55), 'NEW_SEX_CAT'] = 'seniorfemale'
df["NEW_PCLASS_AGE"] = df["PCLASS"] / (df["AGE"].astype(int) + 1)
df["NEW_FARE_AGE"] = df["FARE"] / (df["AGE"].astype(int) + 1)

######################################################
# 2.4 Label Encoding
######################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

######################################################
# 2.5 One-Hot Encoding
######################################################
cat_cols = [col for col in df.columns if (18 >= df[col].nunique() > 2) & (df[col].dtypes == "O")]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

df.head()

######################################################
# 2.6 Rare Encoding
######################################################

# Rare Encoding Analysis. (checking for ratio>0.01 Level)
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "SURVIVED", cat_cols)


# Rare Encoding for ratio<0.01
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df, 0.01)


######################################################
#2.7 Standart Scaler
######################################################
scaled_cols = [col for col in df.columns if col != "SURVIVED"]
scaler = StandardScaler ()
df[scaled_cols] = scaler.fit_transform (df[scaled_cols])

######################################################
# 3. Logistic Regression
######################################################

######################################################
# 3.1 Model
######################################################
df.head()

y = df["SURVIVED"]

X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                     test_size = 0.2,
                                                     random_state=1)

log_model = LogisticRegression().fit(X_train,y_train)

log_model.intercept_ # model constant -> beta value

# coefficients
log_model.coef_

######################################################
# 3.2 Prediction
######################################################
y_pred = log_model.predict(X_train)
y_pred[0:10] # predicted values

######################################################
# 3.3 Success Evaluation
######################################################
y_log_pred = log_model.predict(X_test)
y_log_prob = log_model.predict_proba (X_test)[:, 1]

print(classification_report(y_test, y_log_pred))


#accuracy_score
y_pred = log_model.predict(X_train)
accuracy_score(y_train, y_pred)

# Roc Curve
plot_roc_curve (log_model, X_test, y_test)
plt.title ('ROC Curve')
plt.plot ([0, 1], [0, 1], 'r--')
plt.show ()

# AUC
roc_auc_score (y_test, y_log_prob)