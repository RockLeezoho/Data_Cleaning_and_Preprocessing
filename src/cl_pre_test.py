import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
path = "/Data_Cleaning_and_Preprocessing"

dft = pd.read_csv(path + "/raw_data/test.csv")

dft.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence"], inplace=True)

#fill numeric columns with median
dft["LotFrontage"] = dft["LotFrontage"].fillna(dft["LotFrontage"].median())
dft["MasVnrArea"] = dft["MasVnrArea"].fillna(dft["MasVnrArea"].median())

#no basement
bsmt_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
for col in bsmt_cols:
    dft[col] = dft[col].fillna("None")

#no garage
garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
for col in garage_cols:
    dft[col] = dft[col].fillna("None")
dft["GarageYrBlt"] = dft["GarageYrBlt"].fillna(-1)

#fill Electrical with mode(only 1 value is missing)
dft["Electrical"] = dft["Electrical"].fillna(dft["Electrical"].mode()[0])

#house without fireplace/stone cladding
dft["FireplaceQu"] = dft["FireplaceQu"].fillna("None")
dft["MasVnrType"] = dft["MasVnrType"].fillna("None")

# Duplicates and garbage value treatments
dft = dft.drop_duplicates()

# Outliers treatments
def whisker(col):
    q1, q3 = np.percentile(col.dropna(), [25, 75])
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    return lw, uw

cols_list = ['LotFrontage', 'LotArea', 'MasVnrArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']

#clipping of threshold values
for col in cols_list:
    lw, uw = whisker(dft[col])
    dft[col] = np.where(dft[col] < lw, lw, dft[col])
    dft[col] = np.where(dft[col] > uw, uw, dft[col])

# Encoding of data
cat_cols = dft.select_dtypes(include=["object"]).columns
# print("Columns to be encrypted:", cat_cols.tolist())

#separate columns with few and many values
onehot_cols = [col for col in cat_cols if dft[col].nunique() <= 10]
label_cols = [col for col in cat_cols if dft[col].nunique() > 10]

dft = pd.get_dummies(dft, columns=onehot_cols)

label = LabelEncoder()
for col in label_cols:
    dft[col] = label.fit_transform(dft[col].astype(str))

output_path = path + "/processed_data/processed_test.csv"
dft.to_csv(output_path, index=True)



