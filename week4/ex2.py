import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt

# 데이터 로드
file_path = './wine.csv'
df = pd.read_csv(file_path)

# 결측치 확인
print("=== 결측치 현황 ===")
print(df.isnull().sum())

# 레이블 분포 확인
print("\n=== 레이블 분포 ===")
print(df['Wine'].value_counts())

# 특성과 레이블 분리
X = df.drop('Wine', axis=1).values  # Pandas DataFrame을 NumPy 배열로 변환
y = df['Wine'].values

# 원핫 인코딩 수행 (y 데이터)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y.reshape(-1, 1))

# 훈련 및 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 데이터셋 Shape 확인
print("\n=== 데이터셋 Shape ===")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 하이퍼파라미터 튜닝용 모델 빌더 함수 정의
def build_model(hp):
    model = Sequential()
    # 첫 번째 은닉층: 뉴런 수를 튜닝 (32~128 사이)
    model.add(Dense(hp.Int('units_1', min_value=32, max_value=128, step=32), activation='relu', input_shape=(X_train.shape[1],)))
    # 두 번째 은닉층: 뉴런 수를 튜닝 (16~64 사이)
    model.add(Dense(hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'))
    # 출력층: 클래스 수에 맞게 설정
    model.add(Dense(y_train.shape[1], activation='softmax'))
    
    # 옵티마이저 선택 (Adam 또는 SGD)
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Keras Tuner 설정 (하이퍼파라미터 탐색)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='my_tuning_dir',
    project_name='wine_classification'
)

# 하이퍼파라미터 탐색 수행
tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 최적의 하이퍼파라미터 출력 및 모델 생성
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
print(f"Units in first hidden layer: {best_hps.get('units_1')}")
print(f"Units in second hidden layer: {best_hps.get('units_2')}")
print(f"Optimizer: {best_hps.get('optimizer')}")

# 최적의 하이퍼파라미터로 모델 학습
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 테스트 데이터 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

# 예측 수행 및 결과 확인
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # 예측된 클래스 인덱스
y_true_classes = np.argmax(y_test, axis=1)  # 실제 클래스 인덱스

# 혼동 행렬 출력
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)
