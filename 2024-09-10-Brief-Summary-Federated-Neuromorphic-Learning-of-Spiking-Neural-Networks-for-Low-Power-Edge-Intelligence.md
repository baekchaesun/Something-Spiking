---
layout: post
date: 2024-09-10 10:40:12 +0900 
title: "[Brief Summary] Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence"
---


# Nicolas Skatchkovsky, Hyeryung Jang, and Osvaldo Simeone, "Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence", IEEE ICASSP (2020)

##### 정보통신공학과 202001638 백재선

# I. Introduction

## FL-SNN
- backpropagation 대신 SNN에서 Local update, 주기적으로 기지국 Global update.

## 무엇을 제안하는가?
- 장치 내 학습만으로는 사용가능한 데이터 양이 제한되어있기 때문에 FL을 통한 학습을 사용하여 효율을 끌어올림



# II. System Model

## 각 device는 local datasets를 가지고 학습하고, 이 local datasets 를 device간 주고받지 않고 SNN 모델을 학습시키는 것이 목표

## FL
1. at each iteration, each device computes a local Stochastic Gradient Descent and updates the weight. (Local update)
2. 특정 timestep 마다 모든 devices는 기지국으로부터 centralized averaged parameter을 update (Global update)

## SNN
1. 입출력 모두 binary time sequence로 변환
2. loss function : 확률적 경사 하강법 (SGD)을 이용, 자극을 받아 membrane potential이 상승했을 때 threshold를 넘을 확률 (spike 발생 확률)을 이용 (sigmoid)



# III. FL-SNN : FL with distributed SNNs

## Inter-Stimulus Interval (자극 간 간격)
1. ISI가 짧을수록 EPSP가 커져 spike 발생 가능성이 높음 
2. local datasets 에서 sample을 선택할 때, binary sample time sequence를 얻음.

## FL-SNN은 backpropagation 보다는 local, global feedback signal을 이용하여 학습.

## local and global learning
1. l(t) local learning signal : 현재 신호가 현재 device에 얼마나 효과적인지를 매 반복마다 계산
2. θ(t) global feedback signal : 전체 device의 평균 파라미터를 timestep마다 각 device에 넘겨줌



# IV. Experiments

## FL성능 평가를 위해 2개의 device 중 하나는 samples from class 1, 나머지 하나는 samples from class 7 으로 설정
## timestep (τ)이 작을수록 (자주 global update 할수록) final loss value가 적음.

## major flaws of FL
- large model 파라미터를 계속 전송할 때 생기는 통신 부하
-> 가중치 파라미터 전체를 보내지 않고, 큰 gradient를 기준으로 일부만 전송, 전송되지 않은 가중치는 0으로 설정하는 것이 더 효율적임
