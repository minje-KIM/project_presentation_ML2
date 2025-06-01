# project_presentation_ML2

## 📁 Code Overview

이 레포지토리는 **Kalman Filter-Inspired Visual-Inertial Odometry** 시스템을 구현한 코드 중 일부분입니다. 
주요 구성 요소는 다음과 같이 `Observation Model`과 `Kalman Filter-based VIO Model`로 나뉘며, 각각 별도의 파일에 정의되어 있습니다.

---

### 🔹 `observation_model.py`

> 시각 이미지 및 IMU 센서를 이용해 **관측값 (observation state)** 및 **관측 잡음 공분산 (R matrix)** 을 예측합니다.

#### ▫️ `Inertial_encoder`
- **입력**: IMU 시계열 데이터 (acc + gyro)
- **역할**: 1D CNN으로 IMU 데이터를 고차원 feature vector로 인코딩
- **출력**: (Batch, Seq_len, Feature_dim)

#### ▫️ `Encoder`
- **입력**: 두 연속 이미지 + IMU 시계열
- **역할**: 시각 피처(2D CNN) + 관성 피처(Inertial_encoder) 추출
- **출력**: visual_features, imu_features

#### ▫️ `PolicyNet`
- **입력**: 현재 IMU 피처 + 이전 RNN 상태
- **역할**: 시각 피처를 사용할지 결정하는 Gumbel-softmax 기반 정책 네트워크
- **출력**: binary mask, logits

#### ▫️ `Pose_RNN`
- **입력**: 융합된 피처
- **역할**: LSTM 기반 6DoF pose 회귀 + 관측 잡음 공분산 행렬 \( R \) 추정
- **출력**: 6D pose, \( R \in \mathbb{R}^{6 \times 6} \)

#### ▫️ `DeepVIO`
- **구성**: `Encoder` + `PolicyNet` + `Pose_RNN`
- **역할**: 전체 관측 모델 통합

---

### 🔹 `transition_kalman_vio.py`

> 논문 구조의 핵심인 **전이 모델(Transition Model)** 과 **Kalman 업데이트 과정**을 포함합니다.

#### ▫️ `is_stationary()`
- **역할**: IMU의 변화량을 기준으로 정지 상태 여부 판단
- **용도**: Kalman update에서 움직임 여부를 구분하여 처리

#### ▫️ `TransitionModel`
- **입력**: 이전 상태 \( x_{t-1} \), IMU(acc + gyro), movement classifier 출력 (soft-label)
- **구성**: 3개의 movement 유형(left, straight, right)에 대해 개별 분기
- **출력**: 예측 상태 \( \hat{x}_t \), transition matrix \( A \), noise covariance \( Q \)

#### ▫️ `compute_A_Q()`
- **역할**: NN 출력 벡터를 A(6×6), Q(6×6) 행렬로 변환
- **내용**: 대각 + 비대각 성분 구성 포함 (e.g., x-z, yaw-z 간 공분산)

#### ▫️ `LSTMPrior`
- **역할**: 이전 칼만 게인 \( K \)을 입력받아 LSTM으로 refined Kalman Gain 예측
- **특징**: 시간 일관성 있는 게인 조정을 통해 안정성 향상

#### ▫️ `Kalman_VIO`
- **구성**: `DeepVIO`, `TransitionModel`, `IMUClassifier`, `LSTMPrior`
- **프로세스**:
  1. 관측 모델로부터 \( z_t \), \( R_t \) 추정
  2. IMU 기반 movement 분류기로 soft-label 예측
  3. 전이 모델에서 \( $\hat{x}_t$ \), \( A_t \), \( Q_t \) 예측
  4. Kalman Update 수행하여 최종 상태 \( x_t \) 및 공분산 갱신

#### ▫️ `kalman_update()`
- **역할**: 전이 모델과 관측값을 칼만 필터 수식에 따라 결합하여 상태 추정
- **특징**: 정지 상태에서는 관측값만을 반영하는 특수 처리 포함

#### ▫️ `predict_kalman_gain()`
- **역할**: Kalman Gain \( K = P(P+R)^{-1} \) 계산

---

