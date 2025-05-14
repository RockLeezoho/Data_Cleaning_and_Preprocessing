import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def Processe_data(input_path, output_path):
    df = pd.read_csv(input_path)

    #find missing value
    # print(df.isnull().sum().to_string())

    #find duplicates
    # print(df.duplicated().sum())

    #identifying object value
    # for i in df.select_dtypes(include="object").columns:
    #     print(df[i].value_counts())
    #     print("***"*10)


    # Missing value treatments
    #too many missing values
    df.drop(columns=["Alley", "PoolArea", "Fence", "MiscFeature", "MasVnrType", "FireplaceQu"], inplace=True)

    #fill numeric columns with median
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())

    #no basement
    bsmt_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    for col in bsmt_cols:
        df[col] = df[col].fillna("None")

    #no garage
    garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
    for col in garage_cols:
        df[col] = df[col].fillna("None")
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(-1)

    #fill Electrical with mode(only 1 value is missing)
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    #house without fireplace/stone cladding
    df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
    df["MasVnrType"] = df["MasVnrType"].fillna("None")

    # print(df.isnull().sum().to_string())

    # Duplicates and garbage value treatments
    df = df.drop_duplicates()

    # Outliers treatments (xu ly ngoai le) => Tang do chinh xac, do on dinh
    #Interquartile Range (theo phuong phap IQR) - cat nguong (clipping) du lieu ngoai khoang [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    #only applies to numeric data
    #take whisker threshold
    #Du lieu nam ngoai khoang [lower_bound, upper_bound] duoc xem la outlier
    def whisker(col):
        q1, q3 = np.percentile(col.dropna(), [25, 75])
        #IQR: muc do phan tan cua du lieu giua Q1 va Q3
        iqr = q3 - q1 
        lw = q1 - 1.5 * iqr
        uw = q3 + 1.5 * iqr
        return lw, uw

    cols_list = ['LotFrontage', 'LotArea', 'MasVnrArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']

    #clipping of threshold values
    for col in cols_list:
        #Tinh nguong duoi va nguong tren
        lw, uw = whisker(df[col])
        #Neu gia tri nho hon lw, thay bang lw
        df[col] = np.where(df[col] < lw, lw, df[col])
        #Neu gia tri lon hon uw, thay bang uw
        df[col] = np.where(df[col] > uw, uw, df[col])

    # print(df.columns.tolist())

    # Encoding of data
    #categorical columns will be converted to numeric or binary columns (for One-Hot Encoding).
    #get columns that are object type or contain strings
    cat_cols = df.select_dtypes(include=["object"]).columns
    # print("Columns to be encrypted:", cat_cols.tolist())

    #separate columns with few and many values
    #Tao cot moi cho moi gia tri (khong co thu tu ro rang)
    onehot_cols = [col for col in cat_cols if df[col].nunique() <= 10]
    #Gan so nguyen cho tung gia tri (co thu tu ro rang)
    label_cols = [col for col in cat_cols if df[col].nunique() > 10]

    df = pd.get_dummies(df, columns=onehot_cols)

    label = LabelEncoder()
    for col in label_cols:
        df[col] = label.fit_transform(df[col].astype(str))

    #create table
    df = pd.DataFrame(df) 

    # df.to_csv(output_path, index=False, encoding= 'utf-8')


if __name__ == "__main__":
    path = "D:/PTIT HN/Sem2_2024_25/Nhap_mon_tri_tue_nhan_tao/BTN_AI/Data_Cleaning_and_Preprocessing"  
    Processe_data(path + "/raw_data/test.csv", path + "/processed_data/processed_test.csv")
    Processe_data(path + "/raw_data/train.csv", path + "/processed_data/processed_train.csv")
