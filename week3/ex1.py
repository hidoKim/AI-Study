import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

file_path = './car_evaluation.csv'
df = pd.read_csv(file_path)

print("=== 결측치 현황 ===")
print(df.isnull().sum())

print("\n=== 레이블 분포 ===")
print(df['unacc'].value_counts())

# 레이블 인코딩
label_encoder = LabelEncoder()
encoded_df = df.copy()
for column in df.columns:
    encoded_df[column] = label_encoder.fit_transform(df[column])

# 특성과 레이블 분리
X = encoded_df.drop('unacc', axis=1)
y = encoded_df['unacc']

# 훈련 및 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#dt
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", dt_accuracy)

#rf
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)

#svm
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", svm_accuracy)

#lr
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", lr_accuracy)

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", knn_accuracy)

# Confusion Matrix 계산 및 출력
print("\n=== Confusion Matrices ===")
print("Decision Tree:\n", confusion_matrix(y_test, y_pred_dt))
print("Random Forest:\n", confusion_matrix(y_test, y_pred_rf))
print("SVM:\n", confusion_matrix(y_test, y_pred_svm))
print("Logistic Regression:\n", confusion_matrix(y_test, y_pred_lr))
print("KNN:\n", confusion_matrix(y_test, y_pred_knn))