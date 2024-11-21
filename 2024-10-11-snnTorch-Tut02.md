---
layout: post
date: 2024-10-11 15:41:12 +0900 
title: "[Brief Review] snnTorch-Tut02"
---


https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb

##### 인천대학교 정보통신공학과 202001638 백재선

1. `Hodgkin-Huxley`: 높은 정확도의 생물학적 특징을 가지고 있지만, 현재로서는 복잡성 때문에 사용하기 어려움
2. `LIF`: 생물학적 특징과 실용성 사이 균형적인 모델, 가장 많이 쓰임.
3. `ANN`: 생물학적 특징보단 실용성에 강한 모델, 가중치와 활성화함수를 이용해 학습을 단순화



# I. `snntorch`에서 구현가능한 `LIF` 모델들
- `LIF` 외의 모델도 구현 가능하지만 이번 튜토리얼에서는 `LIF`모델에 대해서만 구현.

### 1. `Lapicque's RC Circuit`: `snntorch.Lapicque`
- RC회로 기반의 간단한 신경 모델
- 제안 당시에는 스파이크 생성 메커니즘이 밝혀지지 않아 회로 형태로 구현됨

### 2. `1st-order model`: `snntorch.Leaky`

### 3. `Synaptic Conductance based neuron model`: `snntorch.Synaptic`
- 시냅스 연결 강도에 따른 전압 변화에 기반한 모델

### 4. `Recurrent 1st-order model`: `snntorch.RLeaky
- `RNN`과 유사하게 동작, 시계열 데이터 처리, LIF의 RNN 버전

### 5. `Recurrent Synaptic Conductance based neuron model`: `snntorch.RSynaptic`
- 시냅스 연결 강도 변화와 뉴런 이전 상태 모두 반영

### 6. `Alpha neuron model`: `snntorch.Alpha`
- 알파함수(EPSP, 입력에 대해 서서히 반응, 일시적인 작은 변화 반영)에 따른 전압 변화에 기반한 모델



# II. `Lapicque's LIF` 모델 간단 구조 및 모델 정의

```
LIF = snntorch.Lapicque(R, C, timestep, threshold, reset_mechanism)  # 많은 하이퍼파라미터 조정 필요

# 구조
def LIF(membrane, input, threshold, timestep, R, C):
	tau = R*C
	spk = (mem > threshold)  # 막전위 > 임계값일 때 스파이크 생성
	mem = mem + (timestep/tau) * (-mem + input*R) - spk*threshold  # 막전위 변화 및 리셋 메커니즘 구현
	return mem, spk
 ```



# III. `Lapicque's LIF` with Pulse Input

### 입력 전류 신호에 대한 뉴런의 반응
### 입력 전류의 진폭과 입력 시간을 조정하며 그래프 변화 관찰
```
cur_in1 = torch.cat((torch.zeros(10, 1), torch.ones(20, 1)*0.100, torch.zeros(170, 1)), 0)  # 진폭 0.1, 20 타임스텝
cur_in2 = torch.cat((torch.zeros(10, 1), torch.ones(10, 1)*0.111, torch.zeros(180, 1)), 0)  # 진폭 0.111, 10 타임스텝
cur_in3 = torch.cat((torch.zeros(10, 1), torch.ones(5, 1)*0.147, torch.zeros(185, 1)), 0)  # 진폭 0.147, 5 타임스텝
cur_in4 = torch.cat((torch.zeros(10, 1), torch.ones(1, 1)*0.500, torch.zeros(189, 1)), 0)  # 진폭 0.5, 1 타임스텝 (Spike)
```
### 입력 전류 진폭을 높이면 타임스텝을 짧게 넣어도 발화할 수 있어서 스파이크 형태의 펄스신호를 입력할 수 있음.



# IV. `Lapicque's LIF`: Firing with Reset

### 리셋 메커니즘을 이용하여 막전위가 임계값을 초과할 시에 한번만 발화하도록 만듦.
### 리셋 메커니즘 미적용시 임계값 초과하는 시점부터 무한히 발화하는 '통제 불가' 문제 발생
### 과분극으로 인한 refactory period도 자연스럽게 구현됨

### 리셋 메커니즘
1. `reset_mechanism = 'subtract'`
- 스파이크 발생 시 임계값만큼 빼는 방식
- 임계값을 초과한 정도도 반영하기 때문에 오차 손실이 비교적 적음.
2. `reset_mechanism = 'zero'`
- 스파이크 발생 시 막전위를 0으로 강제 리셋
- 뉴로모픽 하드웨어에서 스파이크 희소성 촉진 및 전력 효율성
3. 고정 값 리셋 `Reset to Fixed Value`
- 특정 값을 설정해서 그 값으로 강제 리셋하는 방법
4. 적응형 리셋 `Adaptive Reset`
- 뉴런 반응에 따라 다른 값으로 리셋하는 방법



# V. Conclusion

1. 입력 전류의 진폭 조정을 통해 스파이크 형태에 가까운 신호를 생성할 수 있음.
2. 리셋 메커니즘을 통해 뉴런 발화 휴지기와 스파이크 희소성을 만들 수 있음.
3. 실제로는 `Lapicque's LIF` 모델은 거의 사용하지 않음.
- `R`, `C`, `타임스텝`, `임계값`, `리셋 메커니즘` 등 조정할 하이퍼파라미터가 너무 많기 때문.