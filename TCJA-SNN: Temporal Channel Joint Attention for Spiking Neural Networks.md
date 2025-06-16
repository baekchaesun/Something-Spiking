---
layout: post
date: 2024-09-15 10:40:12 +0900 
title: "[Brief Summary] TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks"
---


# R. Zhu et al., "TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks", IEEE Transactions of Neural Networks and Learning Systems (2024)

##### 정보통신공학과 202001638 백재선

# I. Introduction

## 무엇을 제안하는가? - SNN과 attention mechanism의 통합
1. 1-D convolution based two local attention (Temporal, Channel)
2. Cross Convolutional Fusion (CCF) layer : 두 차원 특징 사이에서 연관성을 찾음
3. SNN 모델에 TCJA plug and play block 을 넣음으로써 모델 구조를 건드리지 않을 수 있음
- SNN은 time step간격으로 같은 parameter를 재사용하고 보정하기 때문에 temporal 및 channel 차원 동시에 보정이 가능
- STDP SNN과 temporal-wise attenttion SNN 에서 temporal 차원을 사용했을 때 네트워크 성능이 향상되었음



# II. Related Works and Motivation

## Training SNNs
1. heaviside step function in forward propagation
2. surrogate function (ATan, triangle-like surrogate function)

## Attention mechanism 
1.  the sqeeze and excitation (SE) block
- 네트워크에 쉽게 통합가능
- parameter가 최소로 증가

## Motivation
1. temporal 차원, channel 차원에서 모두 네트워크 성능을 향상 시킬 수 있지만, 기존 SNN 연구들은 모두 둘 중 하나의 데이터로만 처리하며, 공동 특징 추출 능력을 고려하지 않았음.
2. 주변 시간단계와 채널에서 유사한 spike 패턴이 보여질 수 있음
3. attention mechanism으로 parameter가 많아지는 것을 고려해서 1-D convolution을 채택.
4. TCJA module : temporal 정보와 channel 정보를 동시에 처리하여 더 풍부한 특징을 학습



# III. Methodology

## LIF model 
1. surrogate gradient methods 을 이용한 학습이 가능.
2. allows forward and backward propagation along spatial and temporal dimensions.

## temporal-channel joint attention
1. fully connected layer(FC) : # of parameters grows rapidly (with a ratio of T^2 * C^2)
2. 2-D convolutional layer 
- the fixed kernel size 때문에 local area에서만 처리가 가능. 하지만 attention mechanism에서는 local area 뿐만아니라 global correlation을 포착할 수 있어야 함. 
- cnn의 경우 layer stackting으로 receptive field를 확장.

3. TCJA attention mechanism
- receptive field를 확장하면서도 # of parameter를 줄임 (with a raito of T^2 + C^2)
- global cross receptive field
- TLA(temporal local attention), CLA(channel local attention), CCF(공동학습)
	1. Average Matrix by squeezing : 공간적 차원인 Height와 Width 축을 제거하고 temporal, channel 축에 집중 -> 효율적 분석
	2. TLA (Temporal-wise local attention) 
	- 특정 time step의 frame이 가까운 time step의 frame들과 상호작용한다고 주장
	- channel 차원에 대해 timestep별로 1-D convolution 후 average matrix z의 다른 timestep의 출력을 비교하여 feature map을 얻고 누적
	3. CLA (Channel-wise local attention)
	- 특정 channel의 frame이 가까운 channel의 frame들과 상호작용한다고 주장
	- temporal 차원에 대해 Channel별로 1-D convolution 후 average matrix z의 다른 channel의 출력을 비교하여 feature map을 얻고 누적
	4. CCF (Cross-Convolutional Fusion)
	- TLA 후 T matrix와 CLA 후 C matrix 합성으로 fusion matrix F를 만듦.
	- 두 matrix를 내적(dot product)하여 확률표현(sigmoid)
	- 행렬곱을 함으로써 교차항이 생겨 요소들 간 복합적인 관계를 반영하고 frame간 상관관계를 더 강하게 구축

## Training Framework
1. SNN firing is nondifferentiable, so use surrogate function like ATan and triangle for backpropagation.
2. temporal efficient training (TET)-based architectures : TCJA-TET-SNN
3. loss function : spike mean square error (SMSE)
4. the highest firing rate를 가진 뉴런 인덱스를 predicted lable로 간주
5. 1-D convolutional layer와 sigmoid로 간단히 구성되어있어 기존 네트워크에 쉽게 추가 가능
6. 특히, backpropagation에 대한 조정 없이 작동이 가능하여 plug and play module로 활용 가능



# IV. Experiments

## conducted on both event-stream and static datasets for object classification
- CIFAR10-DVS, N-Caltech 101, DVS128 Gesture, Fashion-MNIST, CIFAR10/100

## intergrating approach to convert event stream to frame data
- 분석 효율 및 처리 용이

## CIFAR10-DVS dataset overfitting 방지
- horizontal Flipping, mixup, rolling, rotation, cutout, and shear

## network architecture
1. DVS128 Gesture
- 마지막 2개 pooling layer 앞에 TCJA module 추가
- layer마지막에 1-D average pooling voting layer 추가 (풀링 축소값 중 가장 높은 값 결정)
2. CIFAR10-DVS
- TCJA-TET-SNN와 같은 세팅 -> triangle surrogate function, 마지막 LIF layer를 SMSE loss로 대체
- 기존 TCJA-SNN과 다르게 TCJA module을 첫번째 pooling layer 앞에 추가
3. N-Caltech 101
- 마지막 2개 pooling layer 앞에 TCJA module 추가
- 네트워크가 깊어질수록 더 복잡한 특징을 학습할 수 있음
4. Fashion-MNIST
5. CIFAR10/100
- MS-ResNet 각 블록 아래에 TCJA module 추가
- residual connection으로 vanishing gradient를 방지
- 더 깊은 네트워크에서도 TCJA가 효율적인지 테스트

## Comparison with existing classification SOTA works
- binary spikes를 전체나 부분적으로 사용
- CIFAR10-DVS : 1.7% 향상
- N-Caltech 101 : 1.6% 향상
- DVS128 Gesture : 0.4% 향상 with 3배 적은 timestep (60 vs 20)
- Fashion-MNIST : 0.4% 향상
- CIFAR10 : 0.84 향상
- CIFAR100 : 0.93% 향상
- 전체적으로 좋은 성능

## Comparison with existing image Generation Works
- Fully Spiking VAE (FSVAE)에 TCJA module을 추가하여 테스트
- decoder : TCJA block 이후 temporal 축에서 average output을 계산하는 방법 (시간에 따라 발생하는 출력들을 평균해서 이미지 생성)
- Tempoal Attention Image Decoder (TAID), ESVAE 같은 Generator 모델에 비교했을때, 최고 성능은 아니지만 적응력과 효과 측면에서 비교적 높은 성능을 유지.

## Ablation Study 
- building block을 제거하고 전체 성능에 미치는 효과를 비교해 해당 block의 효과를 얻을 수 있음.
- TLA보다 CLA module이 성능에 더 큰 영향을 줌
- temporal dimension의 firing pattern보다 channel dimention의 firing pattern이 더 중요
- 대부분 SNN에서 timestep 수가 chnnel 수보다 적음(20 vs 128)
- TCJA module이 위 모든 데이터셋에서 다른 모델을 능가했다는 것은 temporal과 channel 차원을 모두 고려하는 것이 더 성능에 좋은 영향을 미친다는 것을 의미
- 모두 고려하기 위한 CCF layer가 이러한 성능을 뒷바침하는 중요한 요소

## Discussion
1. kernel size가 4보다 클 때 전체적은 성능이 감소한다. TCJA module은 서로 가까운 frame간에 강한 상호작용이 있다고 보는데, kernel size가 커지면 원치않는 noise(먼 frame의 영향)를 받게 된다.
2. CCF layer에서 만약 T_ij와 C_ij의 addition 연산을 진행한다면 어떨까?
- 덧셈연산은 두 행렬의 같은 위치 값끼리만 연산을 진행하기 때문에 행렬 내에서 복잡한 상관관계를 포착하기 어려움.
3. training epoch이 증가할수록 더 안정적이게 되고 높은 정확도 level에 수렴

## CCS of Two Orthogonal 1-D Convolution (Temporal - Channel)
1. temporal, channel 연관성 고려
2. cross receptive field를 적용함으로써 복잡하고 다양한 특징을 학습
3. 두 차원 kernel size가 동일하다면, square receptive field를 얻을 수 있음

## TCJA-SNN이 ANN보다 에너지 소모량이 5.26배 낮은 결과 도출됨. (0.0019J vs 0.01J)

## Propagation Pattern (layer-to-layer, step-by-step)
1. step-by-step pattern
- temporal 차원에서 각 time step을 순차적으로 처리하는 방식
- 보통의 SNN이나 RNN에서 사용됨
2. layer-by-layer pattern
- 각 layer의 출력을 한번에 계산하는 방식
- 모든 time step에 대한 첫번째 layer의 출력을 계산하고, 이를 두번째 layer 입력으로 사용을 반복
- CNN, DNN등 일반적인 신경망 구조나 병렬처리가 가능한 환경에서 사용됨
3. TCJA는 layer by layer pattern을 사용하여 학습하는 네트워크지만, step-by-step pattern을 사용할 때도 이점이 존재한다.
- temporal attention 측면에서 TLA module은 몇개의 가까운 time step만 계산하면 된다. TCJA는 kernal size =2일 때 최고 성능을 보여주기 때문에 이 경우 TLA가 step-by-step pattern 학습을 한다면 해당 time step과 나머지 하나 time step만 버퍼링 하면된다.



# V. Conclusion

## 1-D convolution은 frame간 연관성을 만들어냄.
## CCF mechanism은 temporal-channel 차원 간 정보들의 연관성을 만들어냄.
## classification task에서는 압도적인 성능을 보여줌
## generation task에서는 generation 특화모델들에 비교해서 어느정도 경쟁할만한 성능을 보여줌.
## 하지만 TCJA module을 기존 모델에 추가함으로서 생기는 parameter 수 증가는 여전히 문제이다.
