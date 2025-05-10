import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
path = "/Data_Cleaning_and_Preprocessing"

dfn = pd.read_csv(path + "/raw_data/train.csv")

#find missing value
# print(dfn.isnull().sum().to_string())

#find duplicates
# print(dfn.duplicated().sum())

#identifying garbage value
# for i in dfn.select_dtypes(include="object").columns:
#     print(dfn[i].value_counts())
#     print("***"*10)


# Missing value treatments
#too many missing values
dfn.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence"], inplace=True)

#fill numeric columns with median
dfn["LotFrontage"] = dfn["LotFrontage"].fillna(dfn["LotFrontage"].median())
dfn["MasVnrArea"] = dfn["MasVnrArea"].fillna(dfn["MasVnrArea"].median())

#no basement
bsmt_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
for col in bsmt_cols:
    dfn[col] = dfn[col].fillna("None")

#no garage
garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
for col in garage_cols:
    dfn[col] = dfn[col].fillna("None")
dfn["GarageYrBlt"] = dfn["GarageYrBlt"].fillna(-1)

#fill Electrical with mode(only 1 value is missing)
dfn["Electrical"] = dfn["Electrical"].fillna(dfn["Electrical"].mode()[0])

#house without fireplace/stone cladding
dfn["FireplaceQu"] = dfn["FireplaceQu"].fillna("None")
dfn["MasVnrType"] = dfn["MasVnrType"].fillna("None")

# print(dfn.isnull().sum().to_string())

# Duplicates and garbage value treatments
dfn = dfn.drop_duplicates()

# Outliers treatments
#only applies to numeric data
#take whisker threshold
def whisker(col):
    q1, q3 = np.percentile(col.dropna(), [25, 75])
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    return lw, uw

cols_list = ['LotFrontage', 'LotArea', 'MasVnrArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']

#clipping of threshold values
for col in cols_list:
    lw, uw = whisker(dfn[col])
    dfn[col] = np.where(dfn[col] < lw, lw, dfn[col])
    dfn[col] = np.where(dfn[col] > uw, uw, dfn[col])

# print(dfn.columns.tolist())

# Encoding of data
#categorical columns will be converted to numeric or binary columns (for One-Hot Encoding).
#get columns that are object type or contain strings
cat_cols = dfn.select_dtypes(include=["object"]).columns
# print("Columns to be encrypted:", cat_cols.tolist())

#separate columns with few and many values
onehot_cols = [col for col in cat_cols if dfn[col].nunique() <= 10]
label_cols = [col for col in cat_cols if dfn[col].nunique() > 10]

dfn = pd.get_dummies(dfn, columns=onehot_cols)

label = LabelEncoder()
for col in label_cols:
    dfn[col] = label.fit_transform(dfn[col].astype(str))

output_path = path + "/processed_data/processed_train.csv"
dfn.to_csv(output_path, index=True)



