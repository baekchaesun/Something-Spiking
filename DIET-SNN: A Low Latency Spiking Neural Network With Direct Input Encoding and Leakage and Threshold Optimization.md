---
layout: post
date: 2024-09-07 10:40:12 +0900 
title: "[Brief Summary] DIET-SNN: A Low-Latency Spiking Neural Network with Direct Input Encoding and Leakage and Threshold Optimization"
---


# Nitin Rathi and Kaushik Roy, "DIET-SNN: A Low-Latency Spiking Neural Network with Direct Input Encoding and Leakage and Threshold Optimization", IEEE Transactions on Neural and Learning Systems (2023)

##### 정보통신공학과 202001638 백재선

# I. Indroduction

## 현재 SNN 모델의 주요 과제
1. 높은 추론 지연 시간
2. 높은 에너지 사용
3. 긴 훈련 시간
4. 높은 메모리 및 계산 훈련 비용

## 무엇을 제안하는가?
1. backpropagation과 gradient descent를 이용해 올바른 membrane leak and threshold 값 학습
- 활성화 희소성(activation sparsity)과 에너지 효율성 개선
2. 첫번째 컨볼루션 레이어를 spike-generator로 학습
- spiking rate가 weight, membrane leak와 threshold의 함수로 이용됨



# III. DIET-SNN

## Hybrid training method를 적용 (ANN-to-SNN + error backpropagation)

## Direct Input Encoding
첫번째 컨볼루션 레이어 : 특징 추출기이자 spike-generator
-> 첫번째 뉴런이 입력된 픽셀 정보(EPSP)를 누적해서 스파이크를 발생시킴
-> 뉴런의 파라미터가 학습

## 같은 레이어의 모든 뉴런은 같은 membrane leak와 threshold 값을 가짐

## spike는 불연속 미분값을 가지기 때문에 surrogate gradient로 근사하여 미분
- 자극이 threshold에 가까워질수록 gradient 높게 형성



# IV. Experiments

## The three steps
1. begins with training an ANN without bias, batch normalization
-> ANN to SNN 변환 시 최소 손실을 전달
2. ANN is converted to SNN with IF neurons
-> threshold는 각 레이어에서 활성화 전 분포의 95%에 해당하는 값부터 설정
(뉴런이 활성화 되기 전에 입력받은 신호들 중 상위 5%에 해당하는 값들을 threshold로 설정)
3. the converted SNN is trained with error backpropagation to optimize the weights, the membrane leak, and the thresholds of each layers



# V. Energy Efficiency

## spike는 binary하므로 단순 덧셈만 수행 (FP addition)
- ANN의 경우 FP multiplication, FP addition 수행 (MAC)

## neuromorphic hardware are event-driven. (no spike, no waste of energy, and  no compute)

## low leak, high threshold by gradient descent
-> low spike rate
-> energy efficience

## still there's many problems to solve.
1. we did not consider the data movement cost which is dependent on the system architecture and the hard-ware.
2. SNN 학습훈련은 여전히 에너지 효율이 낮음. 고성능 GPU에서도 며칠씩 걸림.
3. 알고리즘과 가속기 모두에서 추가적인 혁신이 요구됨.



# VI. Effect of Direct Input Encoding And Threshold and Leak Optimization

## time steps improving
1. direct input encoding improved to 25 timesteps and spike rate
2. threshold optimization reduced the latency to 15 timesteps and spike rate
3. leak optimization reduced the latency to 5 timesteps and spike rate



# VII. Conclusion

## high sparsity + low inference latency
1. reduces the compute energy by 6-18 x ANN with similar accuracy
2. reduces the number of timesteps by 20-500 x another SNN with similar accuracy
