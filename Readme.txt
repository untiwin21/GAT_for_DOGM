ST-GAT Sensor Fusion for Dynamic Occupancy Grid Map
1. Project Overview (개요)
본 프로젝트는 LiDAR와 Radar(2대) 센서 데이터를 융합하여, 동적 환경에서의 물체 **속도(Velocity)**와 **불확실성(Uncertainty, Sigma)**을 추정하는 Deep Bayesian Sensor Fusion 모델입니다.
이 모델은 Spatio-Temporal Graph Attention Network (ST-GAT) 아키텍처를 기반으로 하며, 추론된 결과는 **DOGM (Dynamic Occupancy Grid Map)**의 파티클 필터 업데이트 단계에 활용되어 기존 물리 기반 필터의 한계(Doppler Ambiguity, Noise)를 극복합니다.
2. System Architecture (시스템 구조)
전체 시스템은 전처리(Preprocessing), 딥러닝 모델(ST-GAT), **DOGM 융합(Fusion)**의 3단계로 구성됩니다.
Raw Data (LiDAR, Radar 1, Radar 2, Odom) 
  --> [Step 1: Odom Transformation & Alignment]
  --> [Step 2: Unified Raw Vector Construction (11-dim)]
  --> [Step 3: Point-wise Embedding (Shared MLP)]
  --> [Step 4: Spatial Encoding (Dense GAT)]
  --> [Step 5: Temporal Aggregation (Cross Attention)]
  --> [Step 6: Output Heads (Velocity & Sigma)]
  --> [Final: DOGM Particle Update]

3. Data Pipeline & Preprocessing
모델은 **과거 4프레임 ($t-4, t-3, t-2, t-1$)**을 입력받아 **현재 ($t$)**의 상태를 예측합니다.
3.1 Odom-based Motion Compensation (동적 보정)
센서 데이터는 측정 당시의 로봇 위치를 기준으로 합니다. 로봇이 이동하기 때문에, 과거 프레임을 단순히 합치면 정적 물체도 움직이는 것처럼 보입니다. 이를 방지하기 위해 모든 과거 데이터를 현재 시점($t-1$ 또는 $t$)의 로봇 좌표계로 변환합니다.
* 수식: $P_{corrected} = T_{rel} \cdot P_{raw}$
* $T_{rel}$: $t_{past}$에서 $t_{curr}$까지의 2D 동차변환행렬 ($3 \times 3$)
   * 회전($\theta$)과 평행이동($x, y$)을 모두 고려하여 좌표를 재계산합니다.
3.2 Unified Raw Vector (입력 벡터 정의)
LiDAR와 Radar는 서로 다른 물리량을 가집니다. 이를 하나의 텐서로 처리하기 위해 Zero-Padding과 Sensor ID를 활용한 11차원 벡터로 통일합니다.
Input Tensor Shape: [Batch, Time(4), Points(1024), Dim(11)]
Index
	Feature Name
	Description
	Source
	비고
	0, 1
	$x, y$
	보정된 2D 좌표
	Common
	$z$축 제외
	2
	$dt$
	상대 시간 ($t_{measured} - t_{curr}$)
	Time
	$0.0 \sim -0.4s$
	3, 4
	$v_{lin}, v_{ang}$
	로봇의 선속도, 각속도
	Odom
	정지/이동 구분용
	5
	$I$
	반사 강도 (Intensity)
	LiDAR
	Radar는 0
	6
	$v_r$
	도플러 속도 (Radial Vel)
	Radar 1 & 2
	LiDAR는 0
	7
	$S$
	신호 대 잡음비 (SNR)
	Radar 1 & 2
	LiDAR는 0
	8, 9, 10
	$d_1, d_2, d_3$
	Sensor ID (One-hot)
	ID
	LiDAR / Radar1 / Radar2
	* Radar 구성: Radar 1 (좌측/전방 등), Radar 2 (우측/후방 등) 두 대의 데이터를 각각 별도의 ID로 구분하여 입력합니다.
* Max Points: 프레임당 최대 포인트는 1024개로 고정하며, 부족할 경우 0으로 패딩(Zero-padding)합니다.
* Batch Size: 32 (설정 가능)
4. Model Architecture: ST-GAT
딥러닝 모델은 입력된 11차원 물리 데이터를 64차원 잠재 특징(Latent Feature)으로 변환하고, 공간적/시간적 문맥을 학습합니다.
Phase 1: Point-wise Embedding (Shared MLP)
* 목적: 이종 센서 데이터를 공통된 잠재 공간(Latent Space)으로 투영.
* 동작: 11차원 벡터에 $1 \times 1$ Convolution (또는 Linear)을 적용.
* 차원 변화: [B, 4, N, 11] $\rightarrow$ [B*4, N, 64]
* 특징: 배치와 시간 차원을 합쳐(Fold) 병렬 처리 효율을 높입니다.
Phase 2: Spatial Encoding (Dense GAT)
* 목적: 이웃한 점들끼리 정보를 교환하여 불확실성을 줄이고 특징을 강화. (예: LiDAR 점이 근처 Radar 점의 속도 정보를 획득)
* 그래프 생성 (k-NN): 특징값($h$)이 아닌, **물리적 좌표($x, y$)**를 기준으로 가장 가까운 $k=16$개의 이웃을 연결.
* Attention Mechanism:
   * 나($h_i$)와 이웃($h_j$)의 특징을 결합(Concat).
   * 학습 가능한 파라미터 $\mathbf{a}$와 내적하여 중요도($\alpha_{ij}$) 산출.
   * $$\alpha_{ij} = \text{Softmax}(\text{LeakyReLU}(\mathbf{a}^T [Wh_i \parallel Wh_j]))$$
* Aggregation: 중요도에 따라 이웃 정보를 가중 합산(Weighted Sum).
* Heads: 4개의 Multi-Head Attention 사용 (Head당 16차원 $\times$ 4 = 64차원 유지).
Phase 3: Temporal Aggregation (Cross Attention)
* 목적: 흩어진 4개의 시간 프레임을 압축하여, 현재 시점($t$)의 상태를 추론.
* Query ($Q$): $t-1$ 시점(가장 최근)의 특징. ("현재 위치와 속도는?")
* Key ($K$) & Value ($V$): $t-4 \sim t-1$ 전체 시퀀스. ("과거의 이동 궤적은 이랬어.")
* 동작: $Q$와 $K$의 유사도를 계산하여, 현재 예측에 유의미한 과거 시점(Trajectory)을 강조.
* 차원 변화: [B, N, 4, 64] $\rightarrow$ [B, N, 64] (시간 차원 소멸)
Phase 4: Output Heads
압축된 문맥 벡터($Z$)를 통해 최종 물리량을 예측합니다.
1. Velocity Head: $(\hat{v}_x, \hat{v}_y)$ 예측. (Cartesian Velocity)
2. Sigma Head: $(\sigma_{pos}, \sigma_{vel})$ 예측. (Aleatoric Uncertainty)
   * Softplus 활성화 함수를 사용하여 항상 양수($>0$)를 보장.
5. Training Strategy (학습 전략)
5.1 Ground Truth (GT) 정의
본 모델은 Self-Supervised (Predictive Coding) 방식으로 학습됩니다.
* 입력: $t-4 \sim t-1$ 프레임.
* 정답 (GT): $t$ 시점의 실제 센서 관측값 (Raw Measurement).
   * 모델은 과거를 보고 현재를 예측하며, 실제 관측된 현재 값과 비교하여 오차를 줄입니다.
   * 속도 GT의 경우, 실제로는 추적(Tracking) 알고리즘이나 미래 프레임($t+1$)을 통해 얻은 Pseudo-GT를 사용할 수 있습니다.
5.2 Loss Function: Gaussian NLL
단순한 MSE가 아닌, **Negative Log Likelihood (NLL)**를 사용하여 불확실성($\sigma$)을 함께 학습합니다.
$$\mathcal{L} = \frac{1}{2} \sum \left( \log(\hat{\sigma}^2) + \frac{(y_{GT} - \hat{y}_{pred})^2}{\hat{\sigma}^2} \right)$$
* 작동 원리:
   * 예측 오차($(y-\hat{y})^2$)가 크면, 모델은 분모인 $\sigma^2$를 키워서 Loss를 줄이려 합니다. (불확실성 증가)
   * 하지만 $\log(\sigma^2)$ 항 때문에 무작정 키울 수는 없습니다. (최적의 균형점 탐색)
* 효과: 센서 노이즈나 고스트 데이터(Ghost Object)에 대해서는 모델이 스스로 **"높은 Sigma"**를 출력하게 되어, DOGM 단계에서 자연스럽게 필터링됩니다.
6. How to Run
6.1 Requirements
* Python 3.x
* PyTorch (CUDA supported)
* NumPy
6.2 Data Preparation
train_stgat.py 실행 시, 같은 디렉토리에 다음 형식의 텍스트 파일이 있어야 합니다. (없으면 더미 데이터 자동 생성)
* LiDAR_DOGM_train1.txt: timestamp x y intensity
* Radar1_DOGM_train1.txt: timestamp x y vr snr
* Radar2_DOGM_train1.txt: timestamp x y vr snr
* Odom_DOGM_train1.txt: timestamp x y theta v_lin v_ang
6.3 Execution
python train_stgat.py

학습이 완료되면 stgat_model.pth 모델 파일이 저장됩니다.
7. Integration with DOGM (최종 융합)
학습된 모델은 DOGM의 C++ 노드와 연동되어 다음과 같이 작동합니다.
1. Infer: 4프레임 데이터를 받아 DL 모델이 $(\hat{v}_x, \hat{v}_y, \sigma_{vel})$ 추론.
2. Fusion: 추론된 값을 FusedMeasurement 구조체로 변환.
3. Update: DOGM 파티클 필터에서 기존 Radial Projection 대신, Cartesian Gaussian Likelihood를 사용하여 확률 갱신.
$$P(z|x) \propto \exp \left( -\frac{(v_{particle} - \hat{v}_{pred})^2}{2 \cdot \hat{\sigma}_{pred}^2} \right)$$