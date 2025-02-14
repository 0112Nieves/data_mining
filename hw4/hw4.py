import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df = pd.read_csv("dataset.csv")
columns_to_input = [
    'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 
    'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 
    'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 
    'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 
    'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 
    'mfcc20'
]
# 確定資料集中有沒有缺失值
# missing_values = df.isnull()
# if missing_values.any().any():
#     print("存在缺失值，位置如下：")
#     for row in range(missing_values.shape[0]):
#         for col in range(missing_values.shape[1]):
#             if missing_values.iloc[row, col]:
#                 print(f"缺失值位置：行 {row + 1}, 列 '{df.columns[col]}'")
# else:
#     print("該資料中沒有缺失值。")
    
# Missing data
# imputer = SimpleImputer(strategy='constant', fill_value=0)
# imputer = SimpleImputer(strategy='mean')
# df[columns_to_input] = imputer.fit_transform(df[columns_to_input])

# Sampling -> 多用在classfication -> 我的的dataset沒有這部分的標籤，故生成一個
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
# y = np.array([0, 1, 0, 1, 0, 1])
# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2)
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2)
# X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y, test_size=0.2, random_state=42)
# print("不設置 random_state:")
# print(X_train_1, X_test_1)
# print(X_train_2, X_test_2)
# print("\n設置 random_state=42:")
# print(X_train_3, X_test_3)

# random_samples = sample_without_replacement(1000, 10, random_state=42)
# print("隨機抽取的樣本:", random_samples)

# Binarize
# binarizer = Binarizer(threshold=2.0)
# binary_data = binarizer.fit_transform(df[columns_to_input])
# print("二元化後的數據:\n", binary_data)
# data = pd.DataFrame({
#     'Color': ['Red', 'Green', 'Blue', 'Green', 'Red'],
#     'Size': ['S', 'M', 'M', 'L', 'S']
# })
# encoder = OneHotEncoder(sparse_output=False)
# encoded_data = encoder.fit_transform(data)
# encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(data.columns))
# print("編碼後的數據:\n", encoded_df)

# Discretization
# X = df[['mfcc18']].values
# discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
# X_binned = discretizer.fit_transform(X)
# print("離散化後的數據:\n", X_binned)

# Standardize
# scaler = StandardScaler()
# df[columns_to_input] = scaler.fit_transform(df[columns_to_input])
# print("標準化後的數據:\n", df.head())
# print("均值:", scaler.mean_)
# print("標準差:", scaler.scale_)
# df[columns_to_input] = scale(df[columns_to_input])

# Normalize -> feature range為指定的min跟Max
# scaler = MinMaxScaler(feature_range=(-1, 1))
# df[columns_to_input] = scaler.fit_transform(df[columns_to_input])
# print("縮放後的數據:\n", df.head())

# Dimension reduction
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(df[columns_to_input])
# pca = PCA(n_components=2)
# data_pca = pca.fit_transform(data_scaled)
# print("降維後的數據:\n", data_pca)

# Feature selection
# data = np.array([[1, 2, 3, 4], 
#                  [1, 2, 3, 4], 
#                  [1, 2, 3, 4], 
#                  [1, 2, 3, 5], 
#                  [1, 2, 3, 6]])
# df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
# selector = VarianceThreshold(threshold=0.1)
# df_reduced = selector.fit_transform(df)
# print("降維後的數據:\n", df_reduced)
# print("被選擇的特徵的索引:", selector.get_support(indices=True))

