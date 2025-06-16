---
layout: post
date: 2024-11-01 15:41:12 +0900 
title: "[Brief Review] snnTorch-Tut05"
---


https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb

##### 인천대학교 정보통신공학과 202001638 백재선



# I. A Recurrent Representation of SNNs
- 일반적으로 `RNN`에서는 `β`(weight decay)를 학습 가능한 파라미터로 취급. (기존 tut3 에서는 하이퍼파라미터)



# II. The Non-Differentiablility of Spikes

### 1. 역전파 학습
- 스파이크는 입력이 임계값을 넘어가는 경우를 제외하면 모든 위치에서 0을 출력하게 함.
- vanishing gradient에 매우 취약하여 `dead neuron problem` 발생. (`dying ReLU`와 비슷한 맥락)
&rightarrow; 일반적인 스파이크 입력은 미분이 불가능하기 때문에 역전파 학습이 불가.

### 2. `dead neuron problem` 해결방안
- 근사 기울기 `surrogate gradient` 이용.
- `Leaky with Surrogate` 모델은 사용자가 직접 정의한 근사 기울기를 사용하여 미분 불가능 문제를 해결.
- `snntorch`의 모든 뉴런 모델은 기본적으로 `ATan` 함수를 이용한 근사 기울기를 자동으로 적용.
- 추가적인 근사 기울기 문서: 'https://brainpy.tech/docs/tutorial_toolbox/surrogate_gradient.html'



# III. BackPropagation Through Time (BPTT)
- `BPTT`는 모든 타임스텝에 대한 기울기를 계산해야함.
- `RNN`의 경우 모든 타임스텝에서 동일한 가중치를 공유해야함.
- `torch.autograd()` 모듈로 기울기 계산을 자동으로 처리.



# IV. Setting up the Loss / Output Decoding
- `snntorch.functional` 모듈은 다양한 손실함수 계산 및 디코딩 방식을 제공.

### 1. 출력 스파이크 해석(디코딩) 방법 (tut1: 입력 인코딩과는 다름.)
- `rate coding`: 가장 높은 발화율을 가진 뉴런을 예측된 클래스로 간주.
- `latency coding`: 가장 먼저 발화하는 뉴런을 예측된 클래스로 간주.

### 2. 손실 계산
- 출력 뉴런 막전위에 `소프트맥스`를 적용하여 클래스의 확률을 계산.
- 타임스텝 별로 발화 정확도가 올라가도록 손실함수 계산 및 수정.
- 최종 손실 계산.



# V. Setting up the Static MNIST Dataset

1. 배치사이즈: 128
2. 이미지 크기 `28*28`로 변환
3. `DataLoader`클래스 이용하여 배치 단위로 로드 및 순회



# VI. Define the Network

1. 입력레이어: 784
2. 은닉레이어: 1000
3. 출력레이어: 10

### Algorithm: Forward Propagation of Spiking neuron with FCN
```
입력: 입력레이어(I), 은닉레이어(H), 출력레이어(O) 개수, 총 타임스텝(T<sub>s</sub>), 입력(X)

1. 네트워크 초기화
|	// I &rightarrow; H &rightarrow; O 형태로 3-Layers FCN 구성

2. 순전파
|	mem1, mem2 정의 및 초기화
|	for t=1 to T<sub>s</sub> do
|	|	cur1 &leftarrow; fc1(X)
|	|	spk1, mem1 &leftarrow; lif1(cur1, mem1)
|	|	cur2 &leftarrow; fc2(cur1)
|	|	spk2, mem2 &leftarrow; lif2(cur2, mem2)

```



# VII. Training the SNN

1. 에포크: 1
2. 손실함수: `CrossEntropyLoss`
3. 최적화함수: `Adam(_, lr=5e-4, betas=(0.9, 0.999))`

### Algorithm: Training Loop of SNN
```
입력: 에포크(E), 입력(X), 타겟(Y), 스파이킹 네트워크 N<sub>s</sub>
for e=1 to E do
|	X<sub>b</sub> &leftarrow; iter(X)   // 입력 데이터 배치단위로 나눔
|
|	for each(X<sub>d</sub>, Y) in X<sub>b</sub>   // X<sub>b</sub>의 데이터와 타겟
|	|	Y<hat> = N<sub>s</sub>(X<sub>d</sub>)   // 순전파
|	|	// 현재 시점까지의 스파이크와 막전위 저장
|	|	Loss &leftarrow; Loss + CrossEntrophy(Y, Y<hat>)   // 현재 시점까지의 손실 계산
|	|	// 역전파를 통한 기울기 계산 및 가중치 업데이트

```

