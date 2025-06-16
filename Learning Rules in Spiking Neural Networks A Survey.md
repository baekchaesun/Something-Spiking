---
layout: post
date: 2024-11-13 10:40:12 +0900 
title: "[Brief Summary] Learning rules in spiking networks: A survey"
---


# [Brief Summary] Zexiang Yi et al., "Learning rules in spiking networks: A survey", Elsvier Neurocomputing (2023)

##### 인천대학교 정보통신공학과 202001638 백재선



# I. Introduction 

## Context: 어떤 종류의 논문인가?
&rightarrow; `SNN`학습 방법의 종합적인 개요
1. `SNN` 기본 개념 및 신경모사 데이터셋 소개
2. 학습 방법들의 특성, 장점, 한계 소개, 다양한 데이터셋에서의 성능 분석
3. 이벤트 기반 시각처리 및 오디오 신호처리 등 `SNN`의 실제 응용 사례
4. 도전 과제 및 미래 연구 방향



# II. Basic concepts of SNNs

## 1. 뉴런 모델
1. LIF (Leaky Integrate-and-Fire)
2. QIF (Quadratic Integrate-and-Fire)
- 막전위가 이차함수 형태로 증가
- 막전위가 임계값에 가까워질수록 더 급격하게 상승
3. IF (Integrate-and-Fire)
4. SRM (Spike Response Model)
- 입력 스파이크와 출력 스파이크에 대한 막전위를 스파이크 응답 커널을 사용하여 모델링
- `Event-driven` 동작
5. PCNN (Pulse Coupled Neural Network)
- `Feeding Input, F`: 기본 입력. 뉴런에 직접적인 자극을 줌.
- `Linking Input, L`: 보조 입력. `F Input`의 자극을 조절.
- `F Input`과 `L Input`은 곱셉 연산으로 조정되어 막전위 형성
- 스파이크 발생 후 막전위를 리셋하지 않고 임계값이 조정 및 누설되어 뉴런 발화 빈도를 조절

## 2. 네트워크 모델

### Feedforward Connections
1. `convolutional connection`
- 국소적인 수용 영역: 뉴런 하나가 입력 데이터의 일부분을 처리하여 복잡한 데이터의 학습이 가능
- 모든 수용 영역에서 가중치가 공유됨
2. `local connection`
- 국소적인 수용 영역
- 각 수용 영역이 고유한 가중치를 가짐 (생물학적으로 더 타당)
3. `full connection`
- 추출된 특징을 분류하는 데에 주로 사용

### Recurrent Connections
1. `self-recurrent connection`
- 자기 순환 연결: 자신의 출력을 다시 입력으로 받아 입력의 변화를 처리
- 시간적 데이터 처리에 용이
2. `lateral connection`
- 같은 계층 내의 뉴런 간 정보 교환으로 안정적인 학습이 가능
- 특징 강화, 상호 보완, 병렬 처리, 노이즈에 강함
3. `feedback connection`
- 피드백 신호를 통해 특정 뉴런 발화를 강화
- `Winner take all, WTA` 메커니즘: 발화 뉴런 외 뉴런은 억제
- 효율적인 정보처리 및 노이즈에 강함

### Residual Connections
- `ANN`에서 발생하는 성능저하 문제 해결
- `Deep SNN`의 핵심 요소

### 세가지 주요 네트워크 모델
1. `Pulse Coupled Neural Network, PCNN`
- 이미지 분할, 융합, 텍스처 검색 등 이미지 처리 작업에 유용
2. `HMAX`
- 합성곱 연산과 서브 샘플링을 통해 계산 효율성을 높이고 일반화된 특징 학습
- 레이어가 깊어질수록 복잡한 특징 학습
- `STDP`를 적용하여 `Spiking HMAX`로 학습이 가능
3. 딥러닝 네트워크 모델
- `SLN`, `MLP`, `RNN`, `CNN` 등 딥러닝에서 널리 사용되는 모델
- `VGG`, `ResNet` 등 신경망 아키텍처

## 3. Synaptic plasticity
- hebbian learning: "Neurons that fire together, wire together."
- 시냅스 이전 뉴런 발화 시간과 이후 뉴런 발화 시간이 시냅스 얼마나 강화할지 (혹은 약화할지) 결정
- 반복적으로 같은 자극을 받을수록 반응 지연이 감소
- `neuromodulator`: 신경조절물질. 시냅스 강화와 약화, 뉴런 발화 시간 차이 기준을 알맞게 조정

## 4. Neural coding
- `Rate Coding`, `Temporal Coding`, `Direct Coding`
- `Direct Coding` 아날로그 신호를 스파이크 트레인으로 변경하여 계산 지연이 줄어듦.



# III. Neuromorphic datasets

- 기존 `ANN`중심의 데이터셋은 `SNN`의 시간 기반 데이터 처리 능력을 활용할 수 없음.
- DVS 카메라를 이용한 `DVS, Dynamic Vision Sensor`: 밝기변화(event-based) (변화만 빠르게 감지, 충돌 회피 시스템)
- `ATIS, Asynchronous Time-based Image Sensor`: 밝기변화 + 절대 광 강도 정보 (주변 조명이 너무 밝거나 어두운 경우 대처)
- `DVS-converted`: `MNIST`, `CIFAR-10` 등 기존 데이터셋을 DVS 카메라를 이용해 동적 이미지로 변환한 데이터셋. `N-MNIST`, `CIFAR1-DVS` 등이 있음.
- `DVS-captured`: `DVS128 Geture` 등 실제 움직임을 기록한 데이터셋.

1. `N-MNIST`: 카메라(ATIS 센서)를 움직이며 기존 데이터셋을 관찰
2. `N-Caltech101`: 카메라(ATIS 센서)를 움직이며 기존 데이터셋을 관찰
3. `CIFAR10-DVS`: `CIFAR-10`의 이미지를 이동시키며 기록
4. `DVS128 Gestrue`: 10종의 제스처
5. `ASL-DVS`: 알파벳에 해당하는 손 제스처



# IV. Learning rules in SNNs
- 지도학습 / 비지도학습
- 생물학적 타당성: `STDP`, Widrow-Hoff 규칙, 기타 생물학적 학습 규칙

## 1. STDP-based rules

### 1.1. 비지도 STDP
- 반복된 특징 감지에 강하지만 드문 특징 감지에 불리
1. `stochastic STDP`로 성능 개선, 유연한 학습
2. 불변 객체 인식 작업에 `STDP` 적용하여 변형된 객체 인식
3. `R-STDP`: 강화학습을 통해 드문 특징 감지 성능 향상

### 1.2. 지도 STDP
1. `anti-STDP`를 결합하여 뉴런 발화 타이밍이 정확하지 않을 때 적용
2. 감독 뉴런이 특정 출력 뉴런에 흥분성 자극을 보내 원하는 발화율로 학습
3. `DA-STDP`에 시냅스 스케일링 결합
- `DA-STDP`: 도파민 보상신호를 이용해 `LTP`와 `LTD`를 조절
- 시냅스 스케일링: 입력 강도를 정규화하여 오버피팅 방지

## 2. CNN과 결합한 STDP 기반 SNN
- `CNN`의 특징 추출 기능을 `SNN`에 결합
1. `deep CovDense SNN`
2. `SpiCNN`

## 3. W-H (Widrow-Hoff) 규칙 기반 학습
- `ANN`에 대한 고전적인 학습 규칙으로, 입력과 출력 간 오차를 이용하여 가중치를 조정.
- W-H 규칙은 역전파를 수행하지 않기 때문에 복잡한 작업을 해결하는 심층 신경망에는 적합하지 않음.
1. `ReSuMe`: W-H 규칙을 `SNN`에 맞춘 방법으로, `STDP`와 `Anti-STDP`를 이용해 가중치를 조정.
2. `Perceptron Based Spiking Neuron Learning Rules (PBSNLR)`: 오분류된 시점의 막전위를 학습에 반영하여 가중치 조정.
3. `Spike Pattern Association Neuron (SPAN)`: 입력, 실제 출력, 목표 출력을 아날로그 신호로 변환하여 스파이크 패턴 비교 및 학습을 더 효과적으로 수행.

## 4. 기타 생물학적 학습 기반 학습
- `STDP`는 네트워크에 전반적으로 영향을 주는 전역 신호가 없어 레이어가 많은 심층 신경망 학습이 어려움.
1. `Self-Backpropagation (SBP)`: 신경망의 각 레이어 별로 목표 출력을 설정하고 자체적으로 오차를 계산하여 가중치 조정.
- 계산 비용 절감 및 에너지 효율 향상.
- 각 레이어가 자율적으로 학습하기 때문에 전체 신경망 측면에서는 목표 출력으로 수렴하지 않을 가능성이 있음.
2. `Spike Synchronization Dependent Plasticity (SSDP)`: `STDP`를 기반으로 하면서, 뉴런의 스파이크가 동기적으로 발생했을 때 시냅스 강도를 강화.
- 뉴런들의 스파이크를 동기화함으로써 시계열 데이터 처리 작업에 효율적임.
- 패턴 인식 작업 시, 뉴런들을 동기화함으로써 유사한 특징을 가진 뉴런들이 동기적으로 발화하도록 유도.

## 5. 기울기 기반 학습
- `SNN`의 출력 스파이크 열에 대한 시냅스 가중치의 기울기는 비미분한 특성을 가짐.

### 5.1. 직접 학습
- 목표 함수 최적화에 사용하는 변수에 따라 네가지로 분류됨.
- 우도(likelihood), 전압(voltage), 타이밍(timing), 활성화(activation)

#### 5.1.1. Likelihood gradient
- 임계값을 확률화하여 뉴런 발화를 확률적 이벤트로 모델링.
- 목표 출력이 발생할 가능성을 최대화하도록 학습.
- 확률을 기반으로 계산하기 때문에 계산 비용이 높아 심층 신경망 학습 효율이 떨어짐.

#### 5.1.2. Voltage gradient
- 스파이크 자체는 미비분성을 띠지만, 뉴런 상태(막전위)는 부드럽게 변화함.
&rightarrow; 막전위를 기준으로 실제 출력과 목표 간 오차를 계산하고 최적화 할 수 있음.
1. `Tempotron`: 오분류가 발생한 시점에서의 최대 막전위와 임계값 간 거리를 최소화하며 학습. 이진 분류 작업에 한정.
2. `Multi-Spike Tempotron (MST)`: 기존 `Tempotron`이 '발화/비발화'로 스파이크를 구분했다면, `MST`는 발화 횟수를 이용하기 때문에 다중 분류 작업이 가능.
3. `NormAD`: 발화 시점 막전위의 근사 기울기를 계산하여 학습.
- 발화 시점에만 학습을 집중함으로써 효율적인 학습이 가능.
- 다층 SNN에 유리하기 때문에 심층 신경망에서도 막전위 기반 학습이 가능.
4. `Spiking CNN`: 막전위가 연속성을 가지도록 근사하여 기울기 기반 학습.

#### 5.1.3. Timing gradient
- 뉴런의 발화 타이밍을 기반으로 기울기를 계산하여 학습.
- 발화 타이밍을 조정하는 방식으로 학습이 이루어지기 때문에 에너지 효율적임.
1. `SpikeProp`: 발화 시점에 막전위를 부분 선형함수로 근사하여 기울기 기반 학습. 뉴런 당 한번만 발화하기 때문에 발화 희소성 이용 가능.
2. `Multi-SpikeProp`: 뉴런들이 여러번 발화할 수 있게 하여 더 복잡한 패턴 학습 가능.
3. `Time-To-First-Spike (TTFS)`: `IF`뉴런의 첫 발화 시간이 `ReLU`활성화 함수와 유사하게 작동하는 것을 이용.
- IF 뉴런 첫 발화 시간: 입력 신호가 클수록 빠르게 발화
- ReLU 함수: 입력이 클수록 큰 출력
&rightarrow; 강한 입력은 빠른 발화, 약한 입력은 늦게 발화
4. 최근 연구 동향: 타이밍 기반 `Deep SNN` 학습
- `Rectified Linear PostSynaptic Potential (ReL-psp)`: 발화 전 막전위가 선형적으로 증가하도록 설계
&rightarrow; 타이밍 기반 학습에서 발생하는 `Dying Neuron Prob`이나 `Exploding Grad Prob` 해결.
 
#### 5.1.4. Activation gradient
- 기존 `SNN`은 막전위가 입계값이 되면 0 &rightarrow; 1을 출력하는 '하드 임계값'을 사용.
- 소프트 임계값을 사용하면, 기울기를 부드럽게 만들어 기존 역전파를 잘 사용할 수 있음.
- 소프트 임계값은 기존 이진 출력의 특성을 가지는 하드 임계값에 비해 저전력 하드웨어 구현에 비효율적임.
1. `Continuous-Coupled Neural Network (CCNN)`: `PCNN`에서 사용되는 계단함수를 시그모이드 함수로 대체하여 역전파가 가능, 이미지 분할 작업에서 뛰어난 성능.
2. `Spatio-Temporal Backpropagation (STBP)`: `LIF`를 반복적으로 사용하여 `SNN`이 `RNN`처럼 `BPTT`를 활용해 가중치를 업데이트.

### 5.2. ANN 사전 학습
- `IF`뉴런의 발화율이 `ReLU`함수를 근사할 수 있다는 점을 이용.
- `ANN`을 사전에 학습한 다음, 학습된 가중치를 `SNN`으로 변환하여 적용.
- `SNN`으로 변환 된 이후 가중치를 세밀 조정 할 수도 있음.
- `ANN`의 성능과 `SNN`의 저전력 구동을 활용할 수 있음.
- 현재는 정적 데이터셋에만 적용가능하며, `SNN`의 시간적 특성을 활용하지 못함.
- 최신 연구 동향: 변환 시 발생하는 오류를 줄이는 방법 개발 및 변환된 SNN의 추론 지연 시간 단축 방법 개발.
1. `ReLU`함수에 임계값과 이동값을 추가해 변환 오류 감소.
2. `Calibration`: `SNN`의 파라미터를 `ANN`에 맞추는 보정 방법.
3. 막전위 초기화를 최적화.
- 변환 후 파라미터 미세 조정: 변환된 SNN 후처리 기술.
1. 누출 계수(leak)와 임계값 공동 최적화.

### 5.3. ANN 연계 학습
- `SNN`은 순전파에 사용, `ANN`은 역전파에 사용하여 가중치를 공유.

## 6. 성능 비교
1. 정적 데이터셋
- ANN 사전 학습과 `Activation gradient` 기반 학습 방법이 주로 이용됨.
- ANN 사전 학습이 정확도 면에서 더 우수한 결과를 보이지만, 추론을 위한 타임스템이 더 많이 필요.
2. 뉴로모픽 데이터셋
- 대부분 `Activation gradient` 기반 학습 방법이 이용됨.
- 뉴로모픽 데이터셋의 시간적 특징을 활용할 수 있기 때문.



# V. Application of SNNs
- `SNN`은 시공간 패턴을 처리하는데 이점을 가짐.
- 이벤트 기반 비전 및 오디오 신호 처리에서의 응용 가능성.
1. 이벤트 기반 비전 (Event-based Vision)
- 이벤트 카메라 장점: 낮은 전력 소모, 높은 시간 해상도
- 높은 시간 해상도란?
&rightarrow; 정밀한 시간 간격 구별
&rightarrow; 매우 짧게 발생하는 변화를 정확히 처리
2. 오디오 정보 처리



# VI. Conclusion and future research directions

## 1. 딥러닝에서 배울 점
- `BPTT`와 배치 정규화 같은 기술을 이용하면서 `SNN` 성능이 크게 향상됨.
- `Network Architecture Search (NAS)`: 최적 신경망 구조를 자동으로 설계하는 방법론.
- `SNN`의 시간적 정보 처리 특성과 스파이크 표현을 고려해야함.
- `BPTT`를 사용해 `Deep SNN`을 훈련하는 경우
&rightarrow; 밀집 텐서 데이터를 처리하는 `BPTT` 특성 상 `SNN`의 스파이크 희소성을 이용할 수 없어 효율이 낮음.
&rightarrow; 밀집 텐서 데이터 처리 시 계산이 복잡하고 학습 속도가 느려져 저전력 구동에 불리.

## 2. 뇌 과학으로부터의 영감
- 현재 `SNN`은 `LIF`같은 단일 뉴런 동역학(시간적 처리)에 집중.
- 네트워크 수준에서의 동역학은 거의 탐구되지 않음. ex) 동기진동 (coherent oscillations), 혼돈 (chaos) 등
1. Burst-Dependent Plasticity: 뉴런이 짧은 시간에 여러 번 발화하면, 해당 신호에 더 큰 중요도 부여.
2. Dendritic Spines (수상돌기 가지치기): 불필요한 시냅스 연결을 제거하여 효율성 향상.
3. Lateral Interations: 같은 레이어의 뉴런들이 서로 신호를 주고받음.
- 흥분성 신호: 중요한 신호를 증폭하여 더 명확하게 패턴 인식.
- 억제성 신호: 과도한 발화를 억제하여 안정성 제공.

## 3. 뉴로모픽 시스템을 위한 알고리즘과 하드웨어 간 시너지
- `SNN`은 시공간 특성 모두 활용하는 네트워크로, 기존 하드웨어에서의 시뮬레이션 방식에 적합하지 않음.
1. 스파이크 기반 계산: 저전력 뉴로모픽 하드웨어.
2. 새로운 장치 활용 - memristor: 생물학전 뉴런과 시냅스의 복잡한 동역학을 모방할 수 있음. 
