---
layout: post
date: 2024-11-21 11:03:12 +0900 
title: "[Brief Summary] SpikFormer: When Spiking Neural Network Meets Transformer"
---


# [Brief Summary] Zhaokun Zhou et al., 'SpikFormer: When Spiking Neural Network Meets Transformer', ICLR (2023)

##### 인천대학교 정보통신공학과 202001638 백재선


- 생물학적으로 타당한(biologically plausible) 두 구조의 결합: `SNN` + `Self-Attention`
- `Self-Attention`: 특성 간 의존성 포착
- `Spiking Self-Attention (SSA)` 메커니즘 및 `Spiking Transformer (Spikformer)` 프레임워크 제안.
1. `Spiking Self-Attention (SSA)`
- 스파이크 형태의 `Query`, `Key`, `Value`를 사용.
- 곱셈 연산이 없어서 에너지 효율적임.
2. `Spikeformer with SSA`
- 뉴로모픽 및 정적 데이터셋 모두에서 이미지 분류 작업 성능이 기존 `SNN`을 능가.



# I. Introduction

- SNN: 최근 ANN의 고급 아키텍처를 차용하여 성능 향상을 보임.
- Transformer: 자연어 처리를 위해 설계된 모델. 현재는 다양한 작업에 활용될 수 있음.
> 1. 이미지 분류
> 2. 객체 탐지
> 3. 시멘틱 세그멘테이션
> 4. 저수준 이미지 처리
- Self-Attention: 트랜스포머의 핵심 요소로, 중요한 정보에 집중.

### 기존 Vanilla Self-Attention (VSA) 구성요소: Query, Key, Value
1. 실수 형태의 Query, Key의 내적을 계산하여 행렬 도출.
2. 소프트맥스 함수로 행렬 정규화를 통해 음수 값을 양수로 변환. (지수 연산, 나눗셈 연산 포함)
3. attention 맵 형성. (Value 가중에 사용됨)
&rightarrow; VSA는 전체적으로 계산 비용이 높아 SNN의 특성과는 맞지 않음.

### Spiking Self-Attention (SSA)

- Query, Key, Value가 모두 스파이크 형태로, 0과 1로만 구성됨.
- SNN + Self-Attention의 주요 문제점: 소프트맥스
&rightarrow; 소프트맥스를 통한 정규화 연산 제거
1. 스파이크를 이용한 attention 맵은 자연스럽게 양수 값으로만 이루어져, 양수로의 변환을 위한 소프트맥스가 필요없음.
2. 이진 값과 논리게이트 AND와 덧셈 연산만을 사용하여 계산 효율이 증가.
3. Query, Key, Value간 계산 순서를 바꿀 수 있음.
4. 시퀀스 길이(N)가 특성 차원(d)보다 클 때 계산 복잡성이 크게 감소.

### 무엇을 제안하는가?
1. Spiking Self-Attention (SSA)
> 스파이크 형태의 Query, Key, Value
> 소프트맥스 제거
2. SSA 기반 Spiking Transformer (Spikformer)
3. 직접 학습을 했음에도 불구하고 뉴로모픽 및 정적 데이터셋 모두에서 최신 SNN을 능가. (SNN은 구조적 특성으로 인해 ANN에서 사전 학습된 가중치를 사용할 수 없는 경우가 많음)



# II. Related work

- Self-Attention 및 Vision Transformer
- SNN
1. ANN-to-SNN: 사전 학습된 ANN의 ReLU 활성화 레이어를 스파이킹 뉴런으로 대체하여 변환.
> ReLU 활성화를 정확히 근사하기 위해 많은 타임스텝 필요. 
2. Direct Training: BPTT를 통한 SNN 직접 학습.
> surrogate gradient 사용.



# III. Method

## 1. Overall Architecture

### 1.1. Spiking Patch Splitting (SPS) 모듈을 이용하여 입력 인코딩
1. 입력 이미지 데이터를 프로젝션을 사용하여 스파이크 특징 벡터로 변환.
- 프로젝션 (projection, 선형 투영): 행렬 곱셈을 통해 고차원 벡터(원본 데이터 `x`)를 저차원 벡터(투영된 데이터 `y`)로 변환. `y = Wx`
2. 변환된 스파이크 특징 벡터를 평탄화하고 N개의 스파이크 패치 시퀀스 `x`로 만듦.
- 패치: 입력 이미지를 일정한 크기로 나눈 것.
- 평탄화 (flatten): 데이터를 1차원 벡터로 변환.
&rightarrow; 입력 이미지 데이터를 N개의 1차원 스파이크 패치 시퀀스로 변환.

### 1.2. Relative Position Embedding (RPE) 생성
- Self-Attention은 입력 이미지의 픽셀 및 패치 위치 (텍스트의 경우 단어 순서)에 의존하지 않음을 보완.
- 기존의 Static Position Embedding 방식은 SNN에 사용할 수 없음.
- 스파이크 데이터 특성에 맞춰 상대적인 위치 정보를 생성.
&rightarrow; RPE를 x에 더하여 X<sub>0</sub> 생성.

### 1.3. X<sub>0</sub>를 Spikformer Encoder Block로 전달
- Spikformer Encoder Block 구성: SSA, MLP, 각 블록에 잔차연결.

### 1.4. 예측값 Y 출력
1. Spikformer Encoder Block에서 처리된 특징에 전역 평균 풀링 (GAP) 적용하여 정보 압축.
2. FCL과 CH(Classification Head)에 전달.

## 2. Spiking Patch Spitting (SPS)
- SPS 모듈은 이미지를 스파이크 특징 벡터로 프로젝션하고, 패치로 분할.
- SPS 모듈은 여러 블록으로 구성될 수 있음.
- ConvStem 데이터 전처리 원리를 이용하여 SPS 설계.
1. 합성곱 레이어: 귀납적 편향을 도입하여 초기 학습 속도와 효율을 높임.
2. Max-Pooling
3. SPS 블록 수가 많으면 합성곱 레이어의 출력 채널 수는 각 D, D/2, D/4, D/8... 로, 초기 블럭에서부터 점점 D에 도달하는 형태를 보임.

## 3. Spiking Self Attention (SSA) Mechanism
- Spikformer의 핵심 요소인 Spikformer Encoder 첫번째 구성 요소.
- 두번째 구성 요소: MLP (Multi Layer Perceptron) 블록.

### 3.1. Vanilla Self Attention (VSA)
- 부동소수점 행렬 Query, Key, Value이 학습 가능한 선형 행렬 (W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>)에 의해 프로젝션됨.
- Q = XW<sub>Q</sub>, K = XW<sub>K</sub>, V = XW<sub>V</sub>
- 소프트맥스 활성화 함수를 사용하여 정규화.
- 부동소수점 행렬곱과 소프트맥스 함수는 지수 연산과 나눗셈 연산을 포함하므로 SNN의 저전력 특성과 맞지 않음.
- Query와 Key의 내적에 필요한 연산량은 O(N^2)이며, 메모리 사용량도 O(N^2)임.

### 3.2. Spiking Self Attention (SSA)
- 스파이크 행렬 Query, Key, Value가 VSA처럼 프로젝션 된 후, 배치 정규화가 이루어지고, 스파이크 시퀀스로 변환됨.
- Q = SN<sub>Q</sub>(BN(XW<sub>Q</sub>)), K = SN<sub>K</sub>(BN(XW<sub>K</sub>)), V = SN<sub>V</sub>(BN(XW<sub>V</sub>))
- 음수 값을 만들지 않기 때문에 정규화를 위한 소프트맥스 활성화 함수 제거.
- Q, K, V를 직접 곱하는 방법을 채택하여 관련 있는 특징에 집중.
- Q, K, V가 모두 스파이크 행렬이기 때문에 행렬곱 대신 논리게이트 AND연산과 덧셈 연산만으로 attention 맵 계산이 가능.
- 소프트맥스 함수의 제거와 단순화된 계산으로 연산량과 전력소모가 대폭 줄어듦.
- Q, K, V 간 계산 순서를 변경할 수 있음.
&rightarrow; SSA가 VSA보다 SNN에 적합함.

### 3.3 Experiments
- SSA와 네가지 다른 attention 맵 계산 방식 비교.
- 동일한 Spikformer 사용 및 스파이크 행렬 V 사용.
1. 부동소수점 Q와 K의 곱. (소프트맥스 X)
2. ReLU(Q)와 ReLU(K)의 곱. (음수 &rightarrow; 0)
3. LeakyReLU(Q)와 LeakyReLU(K)의 곱. (음수 유지)
4. VSA와 같은 방식. 부동소수점 Q와 K의 곱. (소프트맥스 O)
5. SSA
- CIFAR10-DVS, CIFAR10 데이터셋에서 SSA가 정확도, 연산량, 전력소모 모든 부분에서 가장 높은 성능을 보임.
- CIFAR100 데이터셋에서는 (4)에 비해 정확도가 0.06%p 떨어지지만 (77.92% vs 77.86%), 연산량과 전력소모 부분에서 훨씬 좋은 성능을 보임. OPs: (FLOPs 6.6M vs SOP 1.3M), P: (30µJ vs 1.17µJ)



# IV. Experiments
- 정적 데이터셋: ImageNet, CIFAR10 , CIFAR100
- 뉴로모픽 데이터셋: CIFAR10-DVS, DVS128 Gesture
- 라이브러리: pytorch, spikingjelly, timm

## 1. Static Datasets Classification
- Spikformer -L -D: L개의 Spikformer 인코더 블록, D차원의 특징 벡터.

### 1.1. ImageNet 분류 성능 비교
- Spikformer-8-384: 파라미터 16.81M, 연산량 6.82G, 전력소모 7.734mJ, 타임스텝 4, 분류 정확도 70.24% 
&rightarrow; 논문 발표 당시 SNN 직접학습 SOTA 모델인 SEW-ResNet-152보다 파라미터, 연산량, 전력소모, 분류 정확도 모든 부분에서 높은 성능을 보임.
- Spikformer-8-512: 파라미터 29.68M, 연산량 11.09G, 전력소모 11.577mJ, 타임스텝 4, 분류 정확도 73.38%
&rightarrow; 비교 SNN학습 모델 중 파라미터, 연산량, 전력소모, 분류 정확도 모든 부분에서 가장 높은 성능을 보임.
&rightarrow; ANN ViT-8-512(80.80%)에 비해 분류 정확도가 떨어지지만, 연산량과 전련소모 부분에서 더 효율적임.
- Spikformer-10-512: 파라미터 36.01M, 연산량 13.67G, 전력소모 13.899mJ, 타임스텝 4, 분류 정확도 73.68%
- Spikformer-8-768: 파라미터 66.34M, 연산량 22.09G, 전력소모 21.477mJ, 타임스텝 4, 분류 정확도 74.81%
&rightarrow; Spikformer 인코더 블록 수 L, 임베딩 차원 D가 늘어남에 따라 성능이 개선됨. 

### 1.2. CIFAR10 분류 성능 비교
- Spikformer-4-384: 파라미터 9.32M, 타임스텝 4, 분류 정확도 95.19%
&rightarrow; TET(94.44%)와 ANN ResNet-19(94.97%)보다 높은 성능을 보임.
- Spikformer-4-384 (400E): 분류 정확도 95.51%로, 에포크 수를 400으로 늘렸을 때 성능 향상.
- ANN ViT-4-384(96.73%)보다 1.54%p 낮은 성능을 보임.

### 1.3. CIFAR100 분류 성능 비교
- Spikformer-4-384: 파라미터 9.32M, 타임스텝 4, 분류 정확도 77.86%
&rightarrow; 큰 데이터셋에서 ANN ResNet-19(75.35%)의 더 큰 성능 향상. 
- Spikformer-4-384 (400E): 분류 정확도 78.21%로, 성능 향상.
- ANN ViT-4-384(81.02%)보다 3.16%p 낮은 성능을 보임.

## 2. Neuromorphic Datasets Classification

### 2.1. CIFAR10-DVS 분류 성능 비교
- Spikformer-2-256 (10 timestep): 분류 정확도 78.9%
- Spikformer-2-256 (16 timestep): 분류 정확도 80.9%
&rightarrow; 비교 SNN 모델 중 최고 성능을 보임.

### 2.2. DVS128 Gesture 분류 성능 비교
- Spikformer-2-256 (10 timestep): 분류 정확도 96.9%
- Spikformer-2-256 (16 timestep): 분류 정확도 98.3%
&rightarrow; 부동 소수점 스파이크 (스파이크 강도)를 사용한 TA-SNN (98.6%)보다 분류 정확도가 0.3%p 낮은 성능을 보임.
&rightarrow; 하지만, 이진 스파이크를 사용하는 Spikformer가 연산량과 전력효율 부분에서 훨씬 효율적일 것으로 보임. (표에서는 연산량과 전력효율을 비교하지 않음)

## 3. Ablation Study

### 3.1. 타임스텝
- 타임스텝이 1인 Spikformer보다 타임스텝이 4인 모델의 정확도가 더 높음.
- 하지만 타임스텝이 낮은 환경에서도 준수한 성능을 보임.

### 3.2. SSA
1. SSA를 VSA로 대체하여 테스트.
2. VSA에서 Value를 부동 소수점으로 변환하여 테스트.
- CIFAR10에서는 SSA 사용 Spikformer가 (1)을 사용한 모델과 (2)를 사용한 모델보다 더 높은 성능을 보임.
- CIFAR100에서는 (1), (2) 두 모델보다 각각 0.06%p,  0.51%p 낮은 성능을 보임.
- ImageNet에서는 모델 (1)보다 0.68%p 높은 성능을 보였고, 모델 (2)보다 0.58%p 낮은 성능을 보임.

### 3.3. 기타 Attention
1. 부동소수점 Q와 K의 곱. (소프트맥스 X)
2. ReLU(Q)와 ReLU(K)의 곱. (음수 &rightarrow; 0)
3. LeakyReLU(Q)와 LeakyReLU(K)의 곱. (음수 유지)
- 위 세 모델은 Q, K, V의 내적 값이 너무 커져서 vanishing surrogate gradient 문제가 발생.
&rightarrow; 내적 값이 클 경우, 발화 여부를 나타내는 활성화 함수가 매우 커짐. (포화 상태)
&rightarrow; 포화 상태의 경우 활성화 함수 기울기가 0에 가까워짐.
&rightarrow; 발화 희소성을 활용하지 못하고 vanising gradient 문제 발생.


# V. Conclusion
- 본 연구에서는 SSA 기반 Spikformer를 제안.
- SSA에서는 소프트맥스를 제거하고 스파이크 형태의 Query, Key, Value의 내적을 통해 효율적인 계산을 수행.
- 제안된 모델은 SNN 직접 학습을 했음에도 불구하고 뉴로모픽 및 정적 데이터셋 분류 문제에서 최신 SNN 모델을 능가. 