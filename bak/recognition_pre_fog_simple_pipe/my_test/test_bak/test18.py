from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from pandas import read_csv
num_cols = """A100_accelerometer_x	A100_accelerometer_y	A100_accelerometer_z	A100_gyroscope_x	A100_gyroscope_y	A100_gyroscope_z
A100_accelerometer_x	A100_accelerometer_y	A100_accelerometer_z	A100_gyroscope_x	A100_gyroscope_y	A100_gyroscope_z"""
num_cols = num_cols.split()

df = read_csv(r'../data/SignalMarker/pd000001==20161223_1_1.csv')
colt = ColumnTransformer([('standardscaler', StandardScaler(), num_cols),('maxabsscaler', MaxAbsScaler(), num_cols)], remainder='drop')
colt.fit_transform(df)