# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist  # 수정된 부분

# 1. MNIST 데이터 로드 및 확인
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print("Training data shape:", train_X.shape)  # (60000, 28, 28)
print("Test data shape:", test_X.shape)      # (10000, 28, 28)

# 데이터 시각화 (첫 번째 이미지)
plt.imshow(train_X[0], cmap='gray')
plt.title(f"Label: {train_y[0]}")
plt.show()

# 2. 머신러닝 작업: SVM, DT, RF, LR, KNN
# (1) 데이터 전처리 - 1차원 벡터로 변환 및 정규화
train_X_flat = train_X.reshape(train_X.shape[0], -1) / 255.0
test_X_flat = test_X.reshape(test_X.shape[0], -1) / 255.0

# (2) 모델 정의 (일부 데이터로 학습)
models = {
    "SVM": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# (3) 모든 머신러닝 모델 학습 및 평가 + Confusion Matrix
confusion_matrices = {}
print("\n=== 머신러닝 모델 ===")
for name, model in models.items():
    model.fit(train_X_flat[:10000], train_y[:10000])  # 10,000개 샘플로 학습
    predictions = model.predict(test_X_flat)
    accuracy = accuracy_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions)
    confusion_matrices[name] = cm
    print(f"{name} Test Accuracy: {accuracy:.4f}")

# (4) Confusion Matrix 출력
print("\n=== 머신러닝 모델 Confusion Matrix ===")
for name, cm in confusion_matrices.items():
    print(f"\n** {name} **")
    print(cm)

# 3. 딥러닝 작업: CNN
# (1) 데이터 전처리 - 채널 추가 및 정규화
train_X_cnn = train_X.reshape(train_X.shape[0], 28, 28, 1) / 255.0
test_X_cnn = test_X.reshape(test_X.shape[0], 28, 28, 1) / 255.0

# 레이블 원핫 인코딩
train_y_cnn = to_categorical(train_y, num_classes=10)
test_y_cnn = to_categorical(test_y, num_classes=10)

# (2) CNN 모델 정의
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# (3) CNN 모델 학습 및 평가
print("\n=== 딥러닝: CNN ===")
cnn_model.fit(train_X_cnn, train_y_cnn, epochs=5, batch_size=32)

# (4) CNN Confusion Matrix 계산
cnn_pred_probs = cnn_model.predict(test_X_cnn)
cnn_pred = np.argmax(cnn_pred_probs, axis=1)
cnn_cm = confusion_matrix(test_y, cnn_pred)

print("\n=== CNN Confusion Matrix ===")
print(cnn_cm)
