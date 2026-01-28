import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
full_data = pd.concat([train, test], sort=False)


full_data.drop(['Id'], axis=1, inplace=True)


full_data['Electrical'].fillna(full_data['Electrical'].mode()[0], inplace=True)
full_data['LotFrontage'].fillna(full_data['LotFrontage'].median(), inplace=True)
full_data['MasVnrType'].fillna(full_data['MasVnrType'].mode()[0], inplace=True)
full_data['MasVnrArea'].fillna(0, inplace=True)
full_data['GarageYrBlt'].fillna(full_data['YearBuilt'], inplace=True)
full_data['GarageFinish'].fillna("None", inplace=True)
full_data['GarageQual'].fillna("None", inplace=True)
full_data['GarageCond'].fillna("None", inplace=True)
full_data['GarageType'].fillna("None", inplace=True)


full_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)


ordinal_map = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,
    'None': 0
}
ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'HeatingQC', 'KitchenQual', 'FireplaceQu',
                    'GarageQual', 'GarageCond']
for feature in ordinal_features:
    if feature in full_data.columns:
        full_data[feature] = full_data[feature].fillna('None')
        full_data[feature] = full_data[feature].map(ordinal_map)


full_data['TotalBathrooms'] = (full_data['FullBath'] +
                                0.5 * full_data['HalfBath'] +
                                full_data['BsmtFullBath'] +
                                0.5 * full_data['BsmtHalfBath'])


full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']


full_data['HouseAge'] = full_data['YrSold'] - full_data['YearBuilt']
full_data['RemodAge'] = full_data['YrSold'] - full_data['YearRemodAdd']


full_data = pd.get_dummies(full_data, drop_first=True)


numeric_feats = full_data.dtypes[full_data.dtypes != "object"].index
skewed_feats = full_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed = skewed_feats[abs(skewed_feats) > 0.75].index
full_data[skewed] = np.log1p(full_data[skewed])


train_processed = full_data[:train.shape[0]]
test_processed = full_data[train.shape[0]:]
y = train['SalePrice']
y = np.log1p(y)
X = train_processed.drop(['SalePrice'], axis=1)
X_test = test_processed.drop(['SalePrice'], axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


print("Train shape:", X_scaled.shape)
print("Test shape:", X_test_scaled.shape)
