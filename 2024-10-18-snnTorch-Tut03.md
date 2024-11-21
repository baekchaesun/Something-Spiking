---
layout: post
date: 2024-10-18 15:41:12 +0900 
title: "[Brief Review] snnTorch-Tut03"
---


https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb

##### 인천대학교 정보통신공학과 202001638 백재선

1. `Lapicque's LIF` 모델은 조정할 하이퍼파라미터가 너무 많음.
2. `1st-order LIF` 모델은 `snntorch.Leaky()`로 정의할 수 있음.
3. 시냅스 가중치 `w`를 학습하여 입력 강도 조절.
&rightarrow; 임계값을 따로 설정하지 않아도 스파이크 희소성을 만들 수 있음.
4. `Lapicque's LIF` 모델에서는 `R`, `C`를 사용하여 지수적 감소를 구현했다면, 이번 모델은 `β` 하나로 구현.



# I. `Lapicque's LIF`와 비교

- `Lapicque's LIF`
```
# Lapicque's LIF
lif = snntorch.Lapicque(R, C, threshold)
```
- `1st-order LIF`
```
# Simplified LIF`
lif = snntorch.Leaky(beta)
```


# II. Reset Mechanism

1. Hard Reset Mechanism
- `subtract`, `zero`, 고정값 리셋, 적응형 리셋 등 특정 값으로 즉시 리셋
2. Soft Reset Mechanism
- 막전위를 덜 급격하게 감소 및 리셋
- 이전 정보를 조금 남겨두어 더 복잡한 패턴 학습 가능
- Hard Reset Mechanism에 β를 곱한 간단한 수식으로 지수적 감소를 이용



# III. FeedForward SNN with 3-FC
```
import torch.nn as nn
import snntorch

# 레이어 초기화
num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.99

fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snntorch.Leaky(beta)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snntorch.Leaky(beta)

# 은닉상태(막전위) 초기화
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

# 시뮬레이션
for step in range(num_steps):
    cur1 = fc1(spk_in[step])
    spk1, mem1 = lif1(cur1, mem1)
    cur2 = fc2(spk1)
    spk2, mem2 = lif2(cur2, mem2)
    
    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)
```



# IV. Conclusion

1. `snntorch.Leaky()`를 이용해 적은 하이퍼파라미터로 `LIF`모델을 만들 수 있음.
2. 시냅스 가중치 `w`를 학습하여 입력 강도를 조절하기 때문에 임계값을 따로 설정하지 않아도 됨.
3. Soft Reset Mechanism은 간단한 수식 추가로 안정적이고 더 복잡한 패턴을 학습할 수 있음.
4. `pytorch`와 `snntorch`를 결합하여 여러 레이어로 구성된 `snn` 모델을 구현할 수 있음.