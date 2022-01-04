# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 1 : Introduction

- More Study Info at : https://jayhong1999.notion.site/AI504-Study-3254606f482e4b7c893a82189275c941

## 1) Intro to AI
### What is AI?
(1) What is AI?
- 기계나 컴퓨터가 인간의 지능을 모방하게 만드는 것
- 이 개념은 Alan Turing이 고안한 만큼, 굉장히 오래된 개념

(2) What is Machine Learning?
- AI의 한 분야
- 특정 task에 적합한 데이터를 기반으로, 통계학적인 개념을 도입시킨 기계를 통해 학습시키는 방법
- 즉, "Data Driven"의 개념
- 데이터 속의 패펀을 파악해서 여러 task를 진행
- Example ) Rule Based, Bayeisan Method, SVM, Boosting etc

(3) What is Deep Learning?
- Machine Learning의 한 분야
- 방대한 양의 데이터와 신경망 네트워크를 이용해, 문제를 해결하는데 좋은 기계를 학습시키는 방법

### Why Deep Learning
(1) SOTA
- State-of-Art : 예술의 경지, 즉 특정 task에서 가장 높은 성능을 보이는 모델
- 많은 SOTA 모델이 DL 기반
- example) Imagenet Classification : 1000장의 이미지를 분류해내는 task
    - Pre-Deep Learning : 2011년도 26% Error
    - Human : 5% Error
    - Deep Learning : 2016년도 GoogLeNet v4 : 3.1% Error

(2) Less Feature Engineering
- 보통의 Machine Learning의 순서 : 데이터 -> 특징 추출 -> 분류 -> 정답 체크
- Deep Learning의 순서 : 데이터 -> 특징 추출 + 분류 -> 정답 체크
- ML에서 특징 추출(Feature Extraction)은 사람이 해야 하는 영역이지만, DL에서는 그것도 기계가 자동으로 학습함
    - 인간의 노력이 적게 들어감
    - Easy to use

### 이게 어떻게 가능하지?
(1) Large Data
- SNS, Collective Intelligence(위키피디아), Mass Media 등에 산재되어있음

(2) 컴퓨팅 파워의 증가
- TPU : Tensor 단위의 계산에 더 특화된 하드웨어 개발 등

### 최신 AI
- 알파고 : 복잡한 패턴의 학습도 가능함을 보여준 사례
- Image Style Transform : 이미지를 다른 스타일로 변환
- NMT : 기계 번역
- Face Recognition : 얼굴 근처에 Bounding Box 만들기

### GPT-3
- 엄청나게 많은 컴퓨팅 파워와 데이터를 통해 학습 (학습에만 약40~50억원)
- Unsupervised Generative LM, 96 층 Transformer 등으로 학습
- Transformer
    - Encoder, Multi-Head Attention, Scaled Dot-product Attention 등 ...
    - 나중에 좀 더 자세하게 다룰 예정

## 2) 수업 소개
### 목표
- 논문에서 소개된 구조를 만들 수 있다.
- 아이디어가 생기면, 해당 모델을 구축할 수 있다.
- AI 기반 연구 분야 탐색할 수 있다.
- 기초적인 내용만 다루기 때문에, 연습을 많이 해야 함

### 수업 구조
- Lecture + Practice
- 2 Assignments
    - Image  Synthesis with GAN (FID Score) 
    - French - Enlgish Translation with Transformer (BLEU Score)
