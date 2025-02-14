import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('data.csv')

# 預處理
X = data.drop(columns=['label'])
y = data['label']

label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

y = LabelEncoder().fit_transform(y)

# 使用MinMaxScaler進行縮放
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNighborsClassifier ()
# clf = KNeighborsClassifier(n_neighbors=5) 
# clf.fit(X_train, y_train) 
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")

# KNighborsRegressor ()
# regressor = KNeighborsRegressor(n_neighbors=5)
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"KNeighborsRegressor 模型均方誤差: {mse:.2f}")
# print(f"KNeighborsRegressor 模型 R² 分數: {r2:.2f}")

# Gaussian Naive Bayes()
# model = GaussianNB(var_smoothing=1e-9)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")

# MultinomialNB()
# model = MultinomialNB(alpha=1.0)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")

# DecisionTreeClassifier()
# model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")

# DecisionTreeRegressor()
# model = DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"均方誤差: {mse:.2f}")
# print(f"R² 分數: {r2:.2f}")

# LinearSVC()
# model = LinearSVC(C=1.0, max_iter=1000, penalty='l2', dual=False, tol=1e-4)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")

# SVC()
# model = SVC(C=1.0, kernel='rbf', gamma='scale', degree=3, probability=True)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")

# MLPClassifier()
# model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42, learning_rate_init=0.001)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")

# MLPRegressor()
# model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"均方誤差 (MSE): {mse:.2f}")
# print(f"R² 分數: {r2:.2f}")

# RandomForestClassifier ()
# clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")

# GradientBoostingClassifier()
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")
