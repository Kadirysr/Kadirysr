import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################
# averagebidding’in, maximumbidding’den daha fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.
############################

# Test Group
test_group = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")  # test group
test_group.head(20)
test_group[["Impression", "Click", "Purchase", "Earning"]].describe().T

# Control Group
control_group = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")  # control group
control_group.head(20)
control_group[["Impression", "Click", "Purchase", "Earning"]].describe().T

############################
# 1. Hipotezleri kur ve yorumlarını yaz.
############################

# H0: M1=M2 (İki grubun toplam satın alma arasında istatistiksel olarak anlamlı farklılık yoktur)
# H1: M1=M2 (...vardır.)

############################
# 2. Varsayımları Kontrol Et
############################

# Normallik Varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:...sağlanmamaktadır.

# Normallik varsayım kontrolü yapılıyor:
test_stat, pvalue = shapiro(
    control_group["Purchase"])  # We do not reject the null hypothesis so "Purchase" are normally distributed.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# (p-value<0.05 ise H0 reddedilir). Burada 0.5891 > 0.05 olduğundan normal dağılım sağlanmaktadır.

test_stat, pvalue = shapiro(
    test_group["Purchase"])  # We do not reject the null hypothesis so "Purchase" are normally distributed.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# (p-value<0.05 ise H0 reddedilir). Burada 0.1541 > 0.05 olduğundan normal dağılım sağlanmaktadır.

# 1.varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik testi)
# 2.varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik) yapılır.

# burada varsayımlar sağlandığından t testi yapılacak.

############################
# 3. Varyans Homojenliği Varsayımı
############################

# H0: Varyanslar Homojendir.
# H1: Varyanslar Homojen değildir.

# levene testini kullanıyoruz.
test_stat, pvalue = levene(control_group["Purchase"], test_group["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# (p-value<0.05 ise H0 reddedilir). p-value>0.05 olduğundan varyanslar homojendir.

############################
# 4. Hipotezin Uygulanması
############################

# 1.varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik testi)
# 2.varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik) yapılır.

# H0: M1=M2 (İki grubun toplam satın alma arasında istatistiksel olarak anlamlı farklılık yoktur)
# H1: M1=M2 (...vardır.)


# varsayımlar sağlandığından bağımsız iki örneklem t testi (parametrik testi) uygulanıyor.
test_stat, pvalue = ttest_ind(control_group["Purchase"], test_group["Purchase"], equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# (p-value<0.05 ise H0 reddedilir). p-value> 0.05 olduğundan reddedilemez ve
# "İki grubun toplam satın alma arasında istatistiksel olarak anlamlı farklılık yoktur" diyebiliriz.

#Özet olarak;
#1. VARSAYIM KONTROLÜÜ
    #1- Normallik Varsayımı kontrolü: shapiro testi uygulandı. Sonuç normal.
    #2- Varyans Homojenliği Varsayımı kontrolü: levene testi uygulandı. Sonuç homojen.
#2. HİPOTEZİN UYGULANMASI
    #1- Normal ve homojen varsayımları sağlandığından t testi (parametrik) uygulandı.
    #NOT: sağlanmasaydı mannwhitneyu testi (non-parametrik test) uygulanacaktı.


# iki yöntem arasında istatistiksel olarak fark olmadığından yeni bir teklif türü olan average bidding'e geçiş yapılabilir.
# Ancak kesin sonuç için diğer verilere dayalı AB testi de yapardım. tüm sonuçlara göre bilgi verirdim.