from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')

X = data.drop(columns=['label'])
y = data['label']

label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

y = LabelEncoder().fit_transform(y)

# KFold()
# scaler = MinMaxScaler()
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# accuracies = []

# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracies.append(accuracy)

# print(f"交叉驗證平均準確率: {sum(accuracies)/len(accuracies):.2f}")

# ShuffleSplit()
# ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# accuracies = []

# for train_index, test_index in ss.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracies.append(accuracy)

# print(f"ShuffleSplit 交叉驗證平均準確率: {sum(accuracies)/len(accuracies):.2f}")

# confusion_matrix()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型準確率: {accuracy:.2f}")
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)

# classification_report()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# report = classification_report(y_test, y_pred)
# print("Classification Report:")
# print(report)

# f1_score()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"F1 分數: {f1:.2f}")

# precision_recall_curve()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# y_scores = model.predict_proba(X_test)
# for i in range(y_scores.shape[1]):
#     precision, recall, thresholds = precision_recall_curve(y_test == i, y_scores[:, i])
#     plt.plot(recall, precision, marker='.', label=f'Class {i}')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve (One-vs-Rest)')
# plt.legend()
# plt.show()