---
layout: post
date: 2024-11-08 15:41:12 +0900 
title: "[Brief Review] snnTorch-Tut06"
---


https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_6_CNN.ipynb

##### 인천대학교 정보통신공학과 202001638 백재선



# I. Surrogate Gradient Descent

## 1. 정의
- 스파이크가 발생하는 임계값에서 스파이크 함수가 비연속적이기 때문에 비미분성을 띰.
- 임계값을 조정하는 방법(Threshold Shifting)과 임계값을 활성화 함수의 중심으로 맞추는 방법(Centering at Threshold)이 있음.
- 기존 `SNN`에서는 스파이크 유무 `[0, 1]`에 대해서만 고려하는 반면, `surrogate grad`를 사용할 땐 발화 확률과 유사하게 변경 이용하여 미비분성을 해결.
- 발화 확률과 유사하게 스파이크 여부를 연속적인 확률 값으로 변환.

### 1.1. Threshold Shifting
- 추가적인 값을 더하거나 빼는 방법으로 임계값을 조정하여 발화 기준값을 바꿈.
- 임계값을 조정하면 발화율을 조절할 수 있어 학습 효율을 향상시킬 수 있음.
- `ATan` 활성화 함수는 임계값을 기준으로 근처의 발화 확률을 부드럽게 만들어줌.

### 1.2. Centering at Threshold
- 임계값을 활성화 함수의 중심으로 맞춤.
- 임계값이 중앙값이 되어 임계값 근처에서 발화 확률이 급격히 높아짐.
- `sigmoid` 활성화 함수의 경우 `σ(0)=0.5`일 때를 임계값으로 설정.
- 막전위가 임계값에 가까워질수록 발화 확률이 0.5에 근접.
- 막전위가 임계값을 초과하면 발화 확률이 급격히 1로 상승.

## 2. 구현
- `snntorch.surrogate()`를 이용하여 대체기울기 함수 정의
```
spike_grad = surrogate.fast_sigmoid(slope)
lif = snn.Leaky(beta, spike_grad)
```
- `fast_sigmoid`: 계산량이 많은 `sigmoid`의 성능 최적화를 위해 경량화된 함수. 계산이 효율적이고 속도가 빠름.



# 2. Define the Network for CSNN (Convolutional SNN) 
- 사용할 합성곱 신경망 아키텍처: 12C5-MP2-64C5-MP2-1024FC10
1. 12C5: 12개의 5x5 합성곱 커널 필터, 12개의 특징 맵 형성.
2. MP2: 2x2 max-pooling 함수.
3. 1024FC10: 1,024개 뉴런을 10개 출력으로 매핑하는 FCN.
&rightarrow; 입력 이미지를 받아 12개의 채널(특징 맵) 생성 후 특징을 유지하면서 크기를 줄임.
&rightarrow; 12채널을 64채널로 확장해 더 많은 특징 추출.
&rightarrow; 특징 맵을 FCL에 전달하여 10개의 출력으로 매핑

```
CSNN = nn.Sequential(nn.Conv2d(1, 12, 5),          // 64x24x24
                   	       nn.MaxPool2d(2),              // 64x12x12                 
                    	       snntorch.Leaky(beta, spike_grad, init_hidden=True),
                    
                    	       nn.Conv2d(12, 64, 5),         // 64x8x8
                    	       nn.MaxPool2d(2),              // 64x4x4
                    	       snntorch.Leaky(beta, spike_grad, init_hidden=True),
                    
                               nn.Flatten(),                      // FCL
                    	       nn.Linear(64*4*4, 10),
                    	       snntorch.Leaky(beta, spike_grad, init_hidden=True)
                    	       )
```
- `.Flatten()`: 각 입력을 일차원 벡터로 변환하여 FCL에 전달할 수 있게 함. 
- 위의 경우 각 이미지가 `64x4x4=1,024`개 특징을 갖는 일차원 벡터로 변환됨.
- 하이퍼파라미터
1. `batch_size = 128`
2. `beta = 0.5`
3. `num_steps = 50`
4. slope of fast sigmoid: `slope = 25`



# 3. Forward propagation을 통한 타임스텝 반복

- 네트워크 클래스 외부에 `forward_prop` 함수를 정의함으로써 타임스텝 반복과 데이터 누적을 쉽게 처리.
- 각 타임스텝의 출력 결과를 `spk_rec`과 `mem_rec`에 저장하여 이후 학습이나 분석에 활용.

### 클래스 내부 `forward` 함수와의 차이점
- `CSNN`은 공간적 특성을 학습하는 `CNN`과 시간적 특성을 학습하는 `SNN`의 두 특징을 모두 활용할 수 있음.
```
for step in range(num_steps):
        spk_out, mem_out = net(data)
```
- `net`클래스 내부의 `forward`함수는 입력 데이터를 타임스텝 한번만 처리.
- 반면, 클래스 외부의 `forward_prop`함수는 다중 타임스텝동안 데이터를 반복으로 처리.
&rightarrow; 시간에 따른 입력 차이를 감지하는 시계열 데이터 패턴 분석에 유용.



# 4. snntorch.Functional을 이용한 손실함수 계산

### 3.1. tut5에서는 뉴런 막전위와 목표 간 차이를 이용하여 CE를 계산.
- 뉴런이 정확한 시점에서 발화하도록 최적화.
```
loss = torch.nn.CrossEntropyLoss()
```

### 3.2. tut6에서는 뉴런의 총 스파이크 수를 이용해 CE를 계산.
- 뉴런의 발화 횟수를 최적화 (발화 희소성).
```
loss = snntorch.Functional.ce_rate_loss()
```



# 5. Additional Experiences
- 다음 네트워크는 tut6에서 진행한 모델보다 더 높은 정확도를 보임. (85% &rightarrow; 97%)
- `beta = 0.5 &rightarrow 0.9` 외의 하이퍼 파라미터 유지.
- 사용한 신경망 아키텍처: 32C5-MP2-64C5-MP2-1024FC256-256FC10
1. 12개의 특징 추출 후 64개 특징을 추출하던 기존 네트워크에서 32개 특징 &rightarrow 64개 특징을 추출하는 것으로 변경.
&rightarrow; 초기 레이어에서 더 많은 특징을 포착할 수 있어 다음 레이어에서 더 구체적이고 복잡한 패턴을 학습할 수 있어 학습 성능이 향상됨.
> 계산 비용이 높아지고 학습 시간이 길어짐.
> 학습할 매개변수가 늘어나서 과적합 문제가 발생할 수 있음.
2. 마지막 FCL을 두개의 레이어 `fc1` (256개 뉴런), `fc2` (10개 뉴런)로 나누어 설정. 
&rightarrow; 모델이 점차적으로 특징을 추상화하게 해서 성능을 끌어올림.
> 단일 레이어에 비해 계산 비용이 높아지고 학습 시간이 길어짐.

 