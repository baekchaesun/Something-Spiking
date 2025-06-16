---
layout: post
date: 2024-10-04 15:41:12 +0900 
title: "[Brief Review] snnTorch-Tut01"
---


# [Brief Review] snnTorch-Tut01: Input Encodings and Visualization

참조: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb

##### 인천대학교 정보통신공학과 202001638 백재선

# I. 데이터 준비 및 로딩

## 데이터셋은 MNIST를 이용
- 60,000개의 학습 데이터에서 서브셋을 생성해 6,000개만을 이용

```
mnist_train = utils.data_subset(mnist_train, subset=10)  # 서브셋
train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)  # DataLoader
```

## 모듈
1.  `import snntorch.utils` : 데이터셋 수정가능
2.  `import torchvision.datasets` : 데이터셋 로드
3.  `import torchvision.transforms` : 이미지 데이터 전처리

## SNN은 시간 기반 데이터 처리를 위한 모델, MNIST는 정적 데이터셋이라는 차이가 있음.
> - method 1. MNIST 데이터를 여러 타임스텝에 입력하여 '정적 동영상' 형태로 입력.
> - method 2. 입력을 스파이크 트레인으로 변환하여 타임스텝 별로 특징, 픽셀의 이산값을 입력.
> - method 1은 snn의 시간적 다양성 표현을 이용하지 못하므로 method 2 이용.



# II. `snntorch`로 코딩 가능한 세가지 스파이크 인코딩

## `snntorch.spikegen` (generator) 모듈이 입력 데이터를 스파이크로 변환
1. Rate coding: `spikegen.rate` -> 스파이크 빈도
2. Latency coding: `spikegen.latency` -> 스파이크 타이밍 (지연)
3. Delta modulation: `spikegen.delta` -> 시간에 따른 입력 변화 (event-driven)



# III. Rate Coding of MNIST

```
from snntorch import spikegen

data = iter(train_loader)  # 입력데이터를 이터레이터 객체로 설정
data_it, targets_it = next(data)  # 다음 배치의 데이터와 레이블

spike_data = spikegen.rate(data_it, num_steps=100, gain=0.25)  # 스파이크 확률을 25%로 설정
```

1. MNIST 데이터의 흰 부분(숫자)은 1, 검은 부분(빈 공간)은 0으로 출력.
2. `spike_data.size()`: [`타임스텝`, `채널(배치크기)`, `채널 수(차원)`, `이미지크기`] = `[100, 128, 1, 28, 28]`

## Summary of Rate Coding

### 뇌는 스파이크 비율로 데이터를 처리하지 않음.
1. 전력 효율이 낮음. (전체 뉴런의 15% 활동만이 설명 가능)
2. 반응 시간이 느림. (평균적으로 반응시간 내에 2개 스파이크 처리 가능)

### 그럼에도 불구하고 스파이크 비율을 사용하는 이유
1. 더 많은 스파이크를 발생시키기 때문에 잡음에 강함.
2. 다른 인코딩 방식들과 함께 뇌에서 작동하는 것으로 예상됨.



# IV. Latency Coding of MNIST

```
data = iter(train_loader)
data_it, targets_it = next(data)

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)  # tau = R*C, 높을수록 느린 발화
```

1. 흰 부분에 가까울수록 일찍 발화, 검은 부분에 가까울수록 늦게 발화
2. 입력값이 클수록 일찍 발화
 
```
입력에 따른 스파이크 타이밍

def convert_to_time(data, tau=5, threshold=0.01):
    spike_time = tau * torch.log(data / (data - threshold))
    return spike_time

raw_input = torch.arange(0,5,0.05)  # [0.00, 0.05, 0.10, ... , 4.90, 4.95] 1차원 텐서
spike_times = convert_to_time(raw_input)
```

## 스파이크 지연 데이터 생성 시 문제점

### 중간 값(회색 부분) 특징이 부족한 문제 발생
- method 1. `tau` 값을 높여 스파이크 시간을 늦추기
- method 2. `linear = True`로 스파이크 시간을 선형화
- 발화 시간이 고르게 분포됨.

### 시뮬레이션 범위를 100 타임스텝으로 설정했음에도 불구하고 모든 발화가 5 타임스텝 안에 발생
- method 1. `tau` 값을 높여 스파이크 시간을 늦추기
- method 2. `normalize = True`로 스파이크가 전체 타임스텝에 걸쳐 발화하도록 정규화
- 타임스텝 낭비 감소

### 대부분의 스파이크가 마지막 타임스텝에서 발생 (정보가 없는 검은 부분)
- method 1. `clip = True`로 중복 특징을 제거하여 뉴런이 한 타임스텝에서 최대 한 번 발화하도록 제한
- 스파이크 희소성과 전력 효율성이 증가

```
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, normalize=True, linear=True, clip=True)
```

## Summary of Latency Coding

1. 스파이크 희소성을 이용하기 때문에 전력 효율성이 좋음
2. 뉴런의 발화 시간 자체가 정보로 사용되기 때문에 빠른 반응이 가능
3. 시간 차이를 이용하기 때문에 정밀도가 높음
4. 뉴런이 최대 한 번 발화하기 때문에 잡음에 약함



# V. Delta Modulation of MNIST

```
def data_to_time_series(loader, sample_factor=2):
    time_series_data = []

    for images, labels in loader:
        flattened_images = images.view(images.size(0), -1)
        sampled_images = flattened_images[:, ::sample_factor]  # 다운샘플링
        
	time_series = torch.cat([img for img in sampled_images], dim=0)
        time_series_data.append(time_series)

    time_series_data = torch.cat(time_series_data)
    return time_series_data

time_series_data = data_to_time_series(train_loader, sample_factor=2)

spike_data = spikegen.delta(time_series_data, threshold=0.8, off_spike=True)  # 입력 값 범위는 [0, 1]
```

1. `snntorch.delta`는 시계열 텐서를 입력받고 타임스텝마다 현재 특징과 다음 특징을 비교함
2. 그 차이가 임계값보다 크면 발화
3. 1차원 시계열 텐서로 변환 시 타임스텝이 너무 길어져 시각화가 어려운 점을 보완
- method 1. `subset = 10`에서 `subset = 10000`으로 수정 
- method 2. `28 x 28`이미지를 `transform.Resize((14,14))`로 축소
- method 3. `sampled_images = flattened_images[:, ::sample_factor=2]`으로 다운샘플링
- `torch.Size = [28 x 28 x 6000]`  ->  `torch.Size = [14 x 14 x 6 / 2]`
- 주의) 단지 스파이크를 시각화하기 위한 작업임.
4. `off_spike = True`로 입력값 감소에 대해서도 스파이크 발생 (기본값: `False`)


## Summary of Delta Modulation

1.  `Biology is Event-Driven.` 망막 광수용체는 시야에 변화가 있을 때만 동작한다는 특징을 이용한 방법
2. 입력값 변화가 없을 땐 발화율이 낮게 형성
3. 반대로 입력값 변화가 급격할 때는 발화 횟수가 변화량을 따라가지 못하여 왜곡이 발생
4. `Event-Driven`이기 때문에 전력 효율성이 좋음



# VI. Visualization

## 1. Spike Raster Plot
- 시간에 따른 뉴런들의 발화 시점을 시각화
- x축: 시간, y축: 뉴런 인덱스

```
spike_data_sample=spike_data_sample.reshape((num_steps, -1))  # reshape: 
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax=fig.add_subplot(111)  # 1행 1열 그래프 배열에서 첫번째 요소를 plot

# 전체 뉴런의 발화
splt.raster(spike_data[:,0].view(num_steps, -1), ax, s=10, c="black")

# 특정 인덱스의 뉴런 발화
splt.raster(spike_data_sample.reshape(num_steps,-1)[:,idx].unsqueeze(1), ax, s=10, c="black", marker="|")
```

## 2. Animation
- 시간 흐름에 따른 입력데이터의 변화를 볼 수 있음

```
spike_data_sample=spike_data[:,0,0]
fig, ax = plt.subplots()
anim=splt.animator(spike_data_sample,fig,ax)
plt.show()
```
