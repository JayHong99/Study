# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 5 Logistic Regression + Neural Network
- More Study Info at : https://jayhong1999.notion.site/Class-5-Logistic-Regression-Neural-Network-289ae60a4e4c49d69969ec8db162de89

## Deep Learning Frame Works
### Theano, Caffe, MXNet 등등
### Tensorflow 1.0
- 모델을 Compile하고 Excute함
- 많은 사람들이 사용
- 빠름
- 개발자에게 유용
### PyTorch
- 자동 Compile
- 상대적으로 느림
- 연구자에게 유용
- 최근에 Tensorflow보다 인기가 높아지는 중
    - arxiv 기준 pytorch가 tensorflow와 비슷하거나 많음

## Logistic Regression
### Logistic regression
- 가장 유명한 통계학 연구 모델
- 예측 변수의 계수 추정으로, 왜 이 Value가 중요하지 등 추정 가능

### Logit
확률 p
- [0~1]사이에 존재

승산 Odds
- p/(1-p)
    - p = 0.8 -> odds : 4
    - p = 0.5 -> odds : 1
- [0 ~ inf] 사이에 존재

Logit = log-odds
- log(p/(1-p))
- [-inf ~ inf]에 존재
    - 값의 스펙트럼이 넓으니 학습에 유용
- logit을 확률에 대해서 해석하면, p는 [0,1] 사이에 존재
    -> Binary 문제에 대한 개별 확률을 의미
    -> Logit을 학습하면 문제 해결 가능
    -> How 학습? => MLE

### Maximum Likelihood Estimate (MLE)
최대우도추정
- 주어진 표본 x를 통해 모집단 모수를 추정

확률함수 => 우도함수 => 로그우도함수 => 음의 로그우도함수(NLL)
- 수식 생략
- 각 Step을 밟아나가면서 확장되는 개념

## Neural Networks
### Neuron
사람의 신경
- 신호를 입력받으면 주변으로 그 신호를 전달

AI의 신경
- Input X와 Bias가 더해저 축적되고, Activation을 거쳐 전달
- Activaiton이 없으면 사실상 곱셈과 다름이 없음 (Non Linearlity를 주는 방법)
- Example) 28*28 이미지 -> 남여 구분 문제
    - 16,000 픽셀 -> 1 노드 (성별)
    - 16,000개의 파라미터와 X를 곱해 나온 값을 합하고, sigmoid 함수를 통과시켜 0.5보다 크면 남성으로 추정
    => 이 경우에는 Logistic Regression 과 동일함

### Hidden Layer
Input과 Output Layer 사이에 숨겨진 Layer를 의미
-Example) Hidden Layer가 4개의 node를 가지고 있다.
    - 이 4가지는 장발, 수염 여부, 화장, 머리띠를 의미한다고 가정
    - 16,000 픽셀로 4가지에 대한 확률을 구하고, 이 4가지로 다시 남여를 구분한다고 해석
    - 하지만 실제로는 해당 4가지는 인간이 정해주는 것이 아닌 컴퓨터가 알아서 학습
- Hidden Layer가 많아지면 더 많은 특징을 잡아냄

### 학습
- Input~ Hidden : 16k * 4
- Hidden ~ Output : 4 * 1
- 약 64,000개의 파라미터를 잘 분류하도록 학습
=> 즉, 분류를 잘 했는지에 대한 Loss를 낮추도록 학습
=> 즉, MLE를 최대화 = NLL을 최소화 하도록 학습

## Backpropagation
### Hidden Layer의 관점에서 살펴보면,
현재 Layer의 Output = f(이전 Layer의 Output * Weight + Bias)
- 즉, 현재 Layer는 이전 layer에 영향을 받고 다음 layer에 영향을 준다.
- 이를 수식으로 해석할 수 있다.
- 이 관계를 모두 펼쳐서 Chain Rule 을 이용해 미분하고, 그 값으로 Loss를 낮추도록 학습하자.

Back Prop
- Loss Function을 Chain Rule을 이용해 분해하고,
- 현재 layer의 정보를 delta로 정의하면
- 현재 layer의 정보는 다음 layer의 미분계수에 현재 layer의 weight matrix의 미분 정보와 input matrix의 미분 정보를 곱한 것과 같다.
- 즉, Top에서부터 시작하면, 순서대로 계산되며 곱해질 수 있다
    => 학습 속도의 향상

특징
- 속도가 빠르다
- 여기서는 FC만 다뤘지만, 모든 NN 에서 적용 가능
- 모든 Layer의 모든 Node에서 미분 계수를 저장하고 있어야 함

## Autograd in Pytorch
### 컴퓨터가 알아서 Backprop 계산
- Autograd
- Theano, Tensorflow, PyTorch는 내장

### 내장 NN 모델을 Directed Acyclic Graph로 표현
- 각각의 node는 수학 계산을 한다.
- 각각의 node는 수학 계산의 미분을 가지고 있다.
- 오차 신호는 output node에서 input node로 흘러 들어간다.