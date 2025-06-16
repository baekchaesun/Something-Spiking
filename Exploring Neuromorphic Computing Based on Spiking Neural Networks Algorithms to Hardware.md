---
layout: post
date: 2024-10-09 10:40:12 +0900 
title: "[Brief Summary] Exploring Neuromorphic Computing Based on Spiking Neural Networks: Algorithms to Hardware"
---


# [Brief Summary] N. Rathi et al., 'Exploring Neuromorphic Computing Based on Spiking Neural Networks: Algorithms to Hardware', ACM Computing Surveys (2023)

##### 인천대학교 정보통신공학과 202001638 백재선



# I. Introduction 

## Context: 어떤 종류의 논문인가?
&rightarrow; 기존 시스템 분석 및 연구 프로토타입 설명을 위한 논문
> 1. SNN을 기반으로 한 신경모사 컴퓨팅의 발전과정을 묘사
> 2. ANN의 한계와 생물 신경망과의 차이를 설명
> 3. SNN의 알고리즘과 특화 하드웨어 연구 양 방향에서 모두 설명 및 결합
> 4. 워크로드 가속화를 위한 메모리 내 시냅스 계산 패러다임 제안

### 아직 SNN 알고리즘들은 `시간` 매개변수를 충분히 활용하지 못함

### 신경형 하드웨어 아키텍처
- SNN의 기본 원칙에서 영감을 얻음
> 1. `Event-Driven` 스파이크 희소성
> 2. 효율적인 병렬 행렬 연산
> 3. 시간적 희소성
- `GPGPU (General Purpose Graphics Processing Units)`, `TPU (Tensor Processing Units)` 등의 워크로드는 ANN 학습에 맞춰진 모델.
- `NVM (Non-Volatile Memory)`: 비휘발성 메모리로, 전원 공급이 없어도 정보를 계속 유지하는 메모리.



# II. Algorithms

## 네 분야의 발전
1. 뉴런 모델링
2. 입력 인코딩
3. 학습 알고리즘
4. 네트워크 아키텍처


## 1. 뉴런 모델링

### 1.1. `IF (Integrate-and-Fire)`, `LIF (Leaky-Integrate-and-Fire)`
- 결정적인(Deterministic) 모델
- 탈분극, 휴지전위, 재분극, EPSP, 임계값, 불응기, 자극이 없을 시 EPSP감소(LIF)

### 1.2. 확률적 뉴런 모델
- 확률적인(Stochastic) 모델
- 자극이 없을 때 스파이크 확률을 설정가능
- 뉴런 발화 시에도 확률적으로 스파이크 생성


## 2. 입력 인코딩

### 2.1. Rate Coding
- 일정 시간 내 스파이크 비율을 이용
- 많은 타임스텝 수가 필요하기 때문에 추론 지연시간이 김
&rightarrow; 높은 성능, 낮은 전력, 시간 효율

### 2.2. Temporal Coding
- 더 적은 스파이크를 이용할 수 있음
- 스파이크 시간 자체를 정보로 이용하기 때문에 추론 지연시간이 짧음
- 훈련 알고리즘이 부족해서 성능이 낮음
&rightarrow; 낮은 성능, 높은 전력, 시간 효율

#### 2.2.1. 종류
1. 로그 시간 코딩 (Logarithmic Temporal Coding): 아날로그 값을 이진수로 인코딩
2. 순위 순서 코딩 (Rank Order Coding): 스파이크의 정확한 시간 대신 발화 순서를 정보로 이용
3. 시간 스위치 코딩 (Temporal Switch Coding): 두 뉴런의 발화 시간 차이를 정보로 이용

### 2.3. Encoding Layer
- 인코딩 함수도 학습의 일부로 할 수 있음
&rightarrow; 학습되는 매개변수를 가질 수 있음

### 2.4. Event-based Sensors
- 이전 입력과 현재 입력의 차이를 이용하여 학습
- 샘플링 속도가 낮게 고정되면 입력 간 차이가 클 때 모션 블러 발생 (빠른 변화)
- 변화가 크지 않은 입력을 다룰 때 전력 낭비 발생

#### 2.4.1. SNN에 적용
1. ANN에서 필요로 하는 복잡한 스파이크 인코딩 단계가 생략됨
2. SNN 층에 있는 이벤트 기반 비동기 처리 (임계치 발화)와 잘 어울림 (입력 간 차이가 클 때 발화)
3. `IBM의 TureNorth`, `Intel의 Loihi` 등 신경모사 하드웨어와 결합하면 높은 에너지 효율을 보임


## 3. 순방향 신경망 (Feedforward Neural Network)
- 순방향 SNN을 위한 비지도 학습, 지도 학습, bio-plausible한 지역 학습에 대해 설명

### 3.1. 비지도 학습 
- 원시 데이터, 데이터 클러스터를 찾는 데 유용
- 변화하는 환경에 적응

#### 3.1.1. STDP (Spike-Timing-Dependent Plasticity)
1. 각 뉴런의 시냅스 강도를 독립적으로 조절
2. 효과적인 시냅스 강화, 비효과적인 시냅스 약화
&rightarrow; 효과적인 시냅스들이 지나치게 강화되고, 비효과적인 시냅스가 끊어져 오버피팅 문제 발생
3. 비효과적인 시냅스를 약화시키고 그 자리에 새로운 정보를 학습하기 때문에 카타스트로픽 망각 문제가 발생 (이전 데이터에 대한 정보 소실)
4. 컨볼루션 레이어를 여러개 두지 못함 (두 개 이상의 레이어는 효율이 떨어짐)
&rightarrow; 이미지 클러스터링, 저수준 특징 추출에는 강하지만 고수준 특징 추출에 약함

#### 3.1.2 Stochastic STDP
1. SNN을 이진 가중치로 학습
2. 기존 STDP보다 메모리 사용량이 적음
3. STDP를 활용한 반지도 학습
> - 비지도 학습으로 가중치 초기화
> - 지도 학습, 기울기 기반 학습을 통해 가중치 미세 조정
> &rightarrow; 빠른 수렴 및 정확도 향상

### 3.2. 지도 학습
- 레이블이 있는 대규모 데이터에 유용
- 예측 값과 레이블 값 차이를 이용해 손실함수를 최소화
- 스파이크는 불연속적이고 미분 불가능하기 때문에 기울기를 그대로 사용할 수 없음
> 1. 훈련된 ANN을 SNN으로 변환하여 추론
> 2. 불연속적인 도함수를 연속함수로 근사하는 대체기울기(surrogate gradient) 사용

#### 3.2.1. ANN-to-SNN conversion
- ANN의 ReLU 뉴런을 SNN의 IF 뉴런으로 매핑
- `시간` 매개변수가 사용되지 않아 추론 지연 시간이 길어짐
- IF 뉴런의 임계값을 조정
> &rightarrow; IF 뉴런의 평균 발화율 함수가 ReLU 뉴런의 활성화 함수와 유사하게 설정

#### 3.2.2. Spike-based backpropagation
- 스파이크의 불연속적인 특징으로 기울기를 그대로 사용할 수 없음
> 1. 대체기울기 (surrogate gradient): 넓은 범위에서 기울기 함수를 근사
> 2. 의사도함수 (pseudo-derivative): 특정 순간에서의 기울기를 근사
- `시간` 매개변수를 이용하여 타임스텝마다 계산을 수행
> &rightarrow; 역전파를 통한 시계열 학습 (BPTT, Backpropagation Through Time)
- 여러번 반복하며 학습하기 때문에 메모리 사용이 큼
> &rightarrow; 소규모 데이터에 유용 

#### 3.2.3. Hybrid learning
- `ANN-to-SNN conversion`과 `Spike-based backpropagation`을 섞은 알고리즘
> - `ANN-to-SNN conversion`의 장점인 '낮은 에포크 수' 사용
> - `ANN-to-SNN conversion`의 단점인 '높은 추론 지연 시간' 해결
> - `Spike-based backpropagation`의 'SNN에서의 경사하강법 (대체기울기, 의사도함수)' 사용
- 시냅스 가중치, 누출 계수와 스파이크 임계값을 모두 학습

#### 3.2.4. Bio-plausible local learning
1. `Spike-based backpropagation`은 높은 정확도를 보이지만, 메모리 사용량이 큼(end-to-end backpropagation)
2. `STDP`는 메모리 효율이 높지만, 정확도와 안정성이 부족함
3. 지역 학습이 `Event-driven` 신경 모사 하드웨어와 더 호환성이 높음
> &rightarrow; end-to-end backpropagation을 사용하지 않으면서 스파이크 기울기를 이용한 지역 학습
- `pre-synaptic activity`, `post-synaptic activity`를 기반으로 신경조절물질(도파민, 세로토닌 등) 분비를 조절하여 가중치 업데이트
- `SNN`의 맥락에서 신경조절물질은 `로컬 오류 신호`
- 모든 레이어가 병렬 학습이 가능

## 4. 순환 신경망 (Recurrent Neural Network)
- `RNN`의 특징인 피드백 연결을 사용
- `SNN`은 EPSP특징 때문에 암묵적으로 재귀적인 특성을 가지고 있음
- `RNN` 학습은 기울기 폭발(exploding gradient)나 기울기 소실(vanishing gradient) 발생 위험이 큼
- `RSNN` 모델을 연구했지만 학습 부분에서 기존 `RNN`보다 성능이 낮음

### 4.1. Spiking LSTM
- `LSTM` 모델에서 계산이 많은 부분을 SNN으로 변환하여 에너지 효율을 끌어올림

### 4.2. LSM (Liquid State Machines)
- `RSNN`에 `희소 연결 저장소(sparsely connected reservoir, liquid)`이 포함된 형태
> 1. 공간적 연결: 흥분성 및 억제성 뉴런들이 액체 저장소 내에서 무작위로 연결돼 입력신호를 고차원 공간으로 투영
> 2. 시간적 연결: 재귀적 연결을 통해 시간 정보를 유지하여 시간에 따른 입력 변화를 반영
- 성능 향상 접근법
> 1. 액체 저장소에서 다양한 행동, 흥분성 수준을 가진 '이질적인 뉴런'을 사용
>> &rightarrow; 응용프로그램 정확도 향상, 복잡성 증가
> 2. 고정된 빠른 연결 `τ_fast`와 학습 가능한 느린 연결 `τ_slow` 두 유형의 시냅스 연결을 이용
> 3. 분할 학습
>> - 여러 작은 액체 저장소가 입력 일부의 패턴 학습하고 글로벌 분류기로 결합
> 4. 깊은 계층적 `LSM`
>> - 여러 레이어의 액체 저장소를 사용
>> - `winner-take-all-encoder`로, 특정 시간 내 입력 뉴런 중 가장 강하게 활성화된 뉴런만 출력을 생성하고, 다른 뉴런들의 출력을 억제
>> - 각 레이어에서 출력된 표현은 `주의(attention) 함수`를 통해 압축 후 분류

## 5. 신경형 API와 라이브러리
- 기존 `ANN`에 최적화된 `PyTorch`, `TensorFlow` 등의 표준 소프트웨어는 `SNN`을 직접 구현할 수 없음
- `LAVA` 소프트웨어 프레임워크는 기존 하드웨어와 신경 모사 하드웨어 양면에서 모델을 구현할 수 있게 함



# III. Applications
1. `SNN`의 재귀적 특성 &rightarrow; 정적 데이터 및 동적 데이터 처리 모두 적합
2. 이산적으로 발생하는 `Event-driven` 데이터 처리 적합
3. 이미지 분류, 제스처 인식, 감정 분석, 의료 응용, 움직임 추정
4. 적대적 공격에 대한 방어 메커니즘

## 1. 이미지 분류
- 픽셀 기반 이미지를 스파이크 트레인으로 변환하여 최소 타임 스텝으로 역전파 학습
> - 프레임 기반 이미지 데이터셋: `MNIST`, `CIFAR10`, `ImageNet`
> - 신경형 데이터셋: `Neuromorphic-MNIST (N-MNIST)`, `CIFAR10-DVS`

## 2. 제스처 인식, 감정 분석, 순차 분류
- 시간적 의존성이 있는 데이터
- `SNN`은 기본적으로 시간적 정보를 사용하는 모델이기 때문에 시간적 의존 데이터 처리에 적합
> - 제스처 인식: `IBM DVS Gestrue`
> - 영화 리뷰 감정 분석: `IMDB`
> - 오디오 분류: `TIDGITS`

## 3. 생의학 응용
- 생물학적 신호는 일반적으로 시간 변화 신호로, SNN에 적합
> - 뇌전도 분석 및 해독 (EEG, Electroencephalogram)
> - 심전도 분석 및 해독 (ECG, Electrocardiogram)
> - 근전도 분석 및 해독 (EMG, Electromyography)
> - 전기피질도 분석 및 해독 (ECoG, Electrocorticography)
>> &rightarrow; 간질 유발 조직이 생성하는 고주파 진동을 `SNN`을 통해 효율적으로 분석 및 해독

## 4. 움직임 추정
- `Event-driven`기반 처리는 인식 및 계획 작업에 유용
> - 고속 이동 중 장애물 감지 및 회피 등 환경 인식 기반 행동
- `ANN`에 비해 `SNN`은 발화 시간 정보를 그대로 사용하기 때문에 이벤트 처리에 적합
- 소규모 학습만을 사용하고, 역동적이고 복잡한 입력을 처리하지 못한다는 한계
> - `spike vanishing`: 다음 뉴런으로 전달되는 스파이크 수가 급격히 줄어 성능 저하
- `Hybrid SNN-ANN architecture`
> - `SNN` 레이어는 이벤트 데이터를 효율적으로 처리
> - `ANN` 레이어는 `end-to-end backpropagation`을 통한 학습 성능 유지
> &rightarrow; 전통적인 컴퓨터 비전의 학습 방법을 기반으로 `SNN`이 사용 될 때, 효율적인 인식 및 행동계획 시스템이 만들어질 수 있음

## 5. 적대적 공격에 대한 방어 메커니즘
- 방어 메커니즘 요소
> 1. 활성화 가지치기 (Activation Pruning)
>> - 불필요한 뉴런이나 작게 발화하는 뉴런을 제거
>> - 노이즈에 민감한 뉴런들이 적대적 공격에 취약
> 2. 입력 이산화 (Input Discretization)
>> - 연속적인 입력을 이산 값으로 변환
>> - 적대적 공격의 미세한 변화에 둔감하게 반응
> 3. 비선형 활성화 함수 (Non-linear Activation Functions)
>> - 입력에 따른 출력이 선형적이지 않아 공격자가 출력을 예측하기 어려움
>> - `ReLU`의 경우 음수 입력을 0으로 출력하기 때문에 일부 공격을 무효화할 수 있음
- `SNN`은 위 방어 메커니즘 요소를 이미 이용하고 있음
> 1. 발화 희소성을 이용하기 때문에 불필요한 뉴런의 발화를 억제
> 2. `SNN`의 입력은 이진값
> 3. 누설(leak) 매개변수를 이용하기 때문에 비선형적인 요소가 포함됨

