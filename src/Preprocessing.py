import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
from sklearn.impute import KNNImputer
path = "D:/PTIT HN/Sem2_2024_25/Nhap_mon_tri_tue_nhan_tao/BTN_AI/Data_Cleaning_and_Preprocessing"

#Read dataset
dft = pd.read_csv(path + "/data/test.csv")
dfn = pd.read_csv(path + "/data/train.csv")

#head
# print(dft.head())
# print(dfn.head())

#tail
# print(dft.tail())
# print(dfn.tail())

#shape
# print(dft.shape)
# print(dfn.shape)

#info
# print(dft.info())
# print(dfn.info())

#find missing value
# print(dft.isnull().sum())
# print(dfn.isnull().sum())

# print(dft.isnull().sum() / dft.shape[0] * 100)
# print(dfn.isnull().sum() / dfn.shape[0] * 100)

#find duplicates
# print(dft.duplicated().sum())
# print(dfn.duplicated().sum())

#identifying garbage value
# for i in dft.select_dtypes(include="object").columns:
#     print(dft[i].value_counts())
#     print("***"*10)
# for i in dfn.select_dtypes(include="object").columns:
#     print(dfn[i].value_counts())
#     print("***"*10)

# Exploratory Data Analysis (EDA)
#descriptive statistics
# print(dft.describe().T)
# print(dfn.describe().T)

# print(dft.describe(include="object"))
# print(dfn.describe(include="object"))

#histogram to understand the distribution
# warnings.filterwarnings("ignore")
# for i in dft.select_dtypes(include="number").columns:
#     sns.histplot(data= dft, x= i)
#     plt.show()
# for i in dfn.select_dtypes(include="number").columns:
#     sns.histplot(data= dfn, x= i)
#     plt.show()

#Boxplot-to-identify Outliers
# warnings.filterwarnings("ignore")
# for i in dft.select_dtypes(include="number").columns:
#     sns.boxplot(data=dft, x=i)
#     plt.show()
# for i in dft.select_dtypes(include="number").columns:
#     sns.boxplot(data=dft, x=i)
#     plt.show()

#scatter plot to understand the relationship
# for i in ['Id', 'LotFrontage', 'LotArea', 'OverallQual',
#        'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
#        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
#        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
#        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
#        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
#        'MiscVal', 'MoSold', 'YrSold']:
#     sns.scatterplot(data= dft, x= i, y= 'MSSubClass')
#     plt.show()
# print(dft.select_dtypes(include= "number").columns)

#correlation with heatmap to interpret the relation and multicolliniarity
# s = dfn.select_dtypes(include= "number").corr()
# plt.figure(figsize= (20, 20))
# sns.heatmap(s, annot= True)
# plt.show()

# Missing Value treatments
#choose the method of imputing missing value like mean, median, mode or KNNIputer
#median
# for i in ["LotFrontage", "MSSubClass", "SaleType"]:
#     dfn[i].fillna(dfn[i].median(), inplace= True)
# print(dfn.isnull().sum())

#the nearest average of that nearest values
# impute = KNNImputer()
# for i in dfn.select_dtypes(include= "number").columns:
#     dfn[i] = impute.fit_transform(dfn[[i]])

#Outliers treatments
#decide whether to do outliers tratment or not, if do how?
def wisker(col):
    q1, q3 = np.percentile(col, [25,75])
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    return lw, uw
wisker(dfn('LotShape'))

for i in ['LotShape', 'LandContour', 'Utilities', 'LotConfig']:
    lw, uw = wisker(dfn[i])
    dfn[i] = np.where(dfn[i] < lw, lw, dfn[i])
    dfn[i] = np.where(dfn[i] > uw, uw, dfn[i])

for i in ['LotShape', 'LandContour', 'Utilities', 'LotConfig']:
    sns.boxplot(dfn[i])
    plt.show()
print(dfn.columns)

# Duplicates and garbage value treatments
#check for duplicate if we have any unique colum in the data set, delete 
#clean the garbage value
dfn.drop_duplicates()

# Encoding of data
#do label encoding and one hot encoding with pd.getdummies
dummy = pd.get_dummies(data= dfn, columns= ["Country", "Status"], drop_first= True)
