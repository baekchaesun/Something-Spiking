---
layout: post
date: 2024-09-27 10:40:12 +0900 
title: "[Brief Summary] Networks of spiking neurons: The third generation of neural network models"
---


# [Brief Summary] Wolfgang Maass (1996), "Networks of spiking neurons: The third generation of neural network models", Pergamon-Elsevier science Ltd

##### 인천대학교 정보통신공학과 202001638 백재선

# 1. Inroduction

## 많은 생물학적 신경망에서 활동전위 (spike) 발생 시간 차이를 이용하여 정보를 저장
## 정해진 시간 간격동안 뉴런이 발화하면 1, 발화하지 못하면 0을 출력
## 수학적으로 생물학적 신경망을 완전히 구현하지는 못했지만, 이전 세대 모델들보다 더 현실적은 출력을 만들어내기에 투자할 가치가 있음
## absolute, relative refractory period 
## 뉴런 발화는 확률적으로 결정

## 무엇을 제안하는가?
- 생물학적 신경망과 유사한 모델로, spike 발생 시간 차이를 이용하는 3세대 인공신경망 SNN을 제안



# 2. Simulation and separation results

## 두 함수를 이용하여 생물학적 뉴런의 응답함수와 임계값 함수를 근사
- A형 함수(계단 함수) : 뉴런의 발화를 잘 표현하고 포착
- B형 함수(4~5개의 선형 구간으로 나눈 함수) : 활동전위처럼 점점 올라가는 것을 표현 가능


# 2-1. Computation of Boolean Functions

## A형 함수 이용
- 가중치(variable weights) 개수보다 spike 발생 시간 차이 기준(variable delays)을 다양하게 설정할 때 모델의 표현력이 더 증가
## CD(concrete boolean) function
- 두 입력 뉴런이 동시에 발화하는 것을 포착
## 입력 노드의 값들이 게이트 g로 전달 될 때 가중합을 계산, 임계값을 넘으면 발화

## Theorem 1.1. CD function을 계산하는 임계값 함수의 하한은 Ω(n/log(n+1))
## Theorem 1.2. CD function을 계산하고 piecewise polynomial activation function (ReLU)을 이용하는 시그모이드 신경망의 하한은 Ω(n^1/2). 또한 piecewise exponential activation function (sigmoid)를 이용할 경우의 하한은 Ω(n^1/4)


# 2-2. Computation of Functions with Analog Input and Boolean Output

## A형 함수 이용
## ED(element distinctness) function
- 입력 뉴런들의 발화 패턴 유사성 포착
## 정해진 시간 간격 내로 동시에 발화해야함 -> 헤어트리거 방지
## 아날로그 입력을 불리언 출력으로 낼 때, 반올림하는 방법 이용 가능

## Theorem 2. ED function을 계산하는 임계값 함수의 하한 Ω(n*log(n))
## Theorem 3. ED function을 계산하는 시그모이드 신경망의 하한 Ω(n)
## Remark 4.1. Theorem 3의 Ω(n) 은 시그모이드 신경망에서 가장 큰 하한이다.
## Remark 4.2. 시그모이드 뉴런 게이트와 임계값 함수 게이트 모두 사용해도 같은 하한을 보인다.

## 수학적으로는 2개의 EPSP가 막전위 임계치를 넘는다고 알려져있지만, 실제로는 6개 EPSP가 필요
## ED tilde(ED 함수 변형) function
- 입력 값 6개가 비슷한 발화 패턴을 보일 때를 포착

## B형 함수 이용 시 동기화된 입력(spike)을 받더라도 이후에는 서로 다른 시간에 발화하는 치명적인 문제 발생 (piecewise linear 부분의 기울기가 동시 EPSP 수에 따라 달라짐)

## Theorem 5. 0~1 사이의 값들을 입력받는 임계값 함수는 B형 함수로 시뮬레이션 할수있다.
## 동기화 모듈 필요
## 보조 뉴런의 추가적인 EPSP


# 2.3. Further Results for Networks of Spiking Neurons of Type B

## 시그모이드 뉴런이 적어도 충분히 근사가 가능
## 시그모이드 함수가 정확하지 않아도 성능에 큰 영향이 없어 linear saturated activation function이라는 근사 함수를 이용
## B형 함수가 꼭 piecewise linear하지 않아도 되고, EPSP에서 선형증가 조금, IPSP에서 선형감소 조금만 필요
## 이론적으로 디지털 입력에 대해 노이즈가 매우 심해도 성능 감소가 적음. 반면 아날로그 입력에 대해서는 성능감소가 있음.



# 3. Conclusion

## CD, ED, ED tilde를 계산하는 모델의 경우 더 적은 뉴런으로 좋은 성능을 낼 수 있음
## 논문 작성 당시 가장 큰 하한인 Ω(n^1/4)에서 Ω(n)으로 가장 큰 하한을 갱신 (ED 사용하는 시그모이드 신경망)