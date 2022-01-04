# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 3 : Basic Machine Learning
- More Study Info at : https://jayhong1999.notion.site/Class-3-Basic-Machine-Learning-adba1913173342e287f9ac26dfc58693

## 1) Categories of ML

- Supervised Learning
    - Input X를 이용해 Output Y를 맞추는 함수(모델) 학습

- Unsupervised Learning
    - 데이터 X만을 이용해 위상 공간이나 분포를 학습

- Reinforcemnet Learning
    - 환경 E에서 행동집합 A가 주어졌을 때, 보상 R을 최대화하도록 학습

## 2) Optimization

- Training Model
    - 모델을 학습시킨다 = Loss를 최소화 한다
    - Loss : 정답과 예측 사이의 차이를 점수화 한 것
    - Classificaiton : Cross-Entropy Loss 등
    - Regression : MSE, MAE 등

- How to Find Globa Minima
    - Global Minima & Local Minima
        - f(x) = x^2 -2x +1의 최소값 = 미분 했을 때 0이 되는 지점
        - f(x)의 차수가 커지면 극소값이 여러개 생김
            - 가장 작은 극소값은 최솟값 = Global Minima
            - 나머지 극소값  = Local Minima라고 부름 (다수 존재)
    - Globa Minima를 찾는 방법
        - 분석적 해결책은 없음
        - 수학적 방법
            - loss가 계속 작아지도록 학습하다보면 만족할만한 theta를 선택
        - Gradient Descent
            - 데이터 X와 Y를 이용해서 Loss를 줄이도록 학습
        - Stochastic Gradinet Descent
            - 데이터 X와 Y의 Subset을 이용해서 Loss를 줄이도록 학습

## 3) Evaluation

- 왜 평가해야하는가?
    - Loss만 가지고는 언제 학습을 멈출지 모름
    - 잘 학습됐다는 기준이 loss로는 직관적임 (0.001이 잘된건지, 0.0001이 잘된거지..)

- 유명한 평가지표
    - 정확도 (다중분류)
    - ROCAUC (이진분류)
    - Precision & Recall (분류)
    - Perplexity (언어 모형)
    - BLEU (언어모형)
    - FID score (GAN)

## 4) Train, Validation, Test Dataset

- Train, Validation, Test split
    - Train : 이 데이터를 이용해서 학습
    - Validation : 이 데이터를 이용해서 잘 학습됐는지 평가
    - Test : 처음보는 데이터 (실무에 사용할거라 생각)에 적합한지 평가

- N-fold Cross Validation
    - 데이터를 N개로 쪼개어, N-1개는 Train으로 사용하고, 1개는 test로 사용
        - 1번 데이터를 test로 사용부터, N번 데이터를 Test로 사용할 때 까지 모델을 총 N 번 학습하고 예측
    - Train, Validaion, Test Split의 단점 보완
    - Lucky Split 혹은 Unlucky Split으로 인한 오해를 방지하고 일반화
    - 데이터가 엄청나게 많으면 Lucky-Unlucky Case가 줄어들기에 사용 안함

## 5) Overfitting & Underfitting

- Overfitting & Underfitting 뜻
    - Underfitting : 모델 학습이 잘 안된 경우 의미
    - Just Fit : 모델이 잘 학습된 경우를 의미
    - Overfitting : 모델 학습이 과하게 된 경우를 의미
- 학습 통제
    - Underfitting : Complexity 증가, Feature 추가, 더 오래 학습
    - OVerfitting : 정규화, 데이터 추가
- 차원의 저주
    - Feature가 선형적으로 증가하면, Feature 공간은 지수적으로 증가
    - 그러면 데이터가 분포해있을 Feature 공간이 너무 넓어짐 (Sparsity)
    - NLP에선 단어 Vocab size의 예시가 있음
- Regularization
    - 모델의 자유도에 제한을 걸기
    - 모델이 학습하는 파라미터에 제한을 걸어 너무 커지거나 작아지지않게 제한

## 6) Models
- Popular Classifiers
    - Logistic Regression : 데이터의 log odds를 계산해 확률로 해석
    - Support Vector Machine : 두 Class 사이의 margin을 최대화하는 방법
    - Decision Tree : 스무고개처럼 데이터를 쪼개는 방식
- Ensembles
    - 여러 모델을 합쳐 성능 향상
    - Bagging
        - 여러 모델에 서로 다른 데이터 학습
        - 여러 모델에 서로 다른 Feature 학습
        => 서로 다른 결과 => Ensemble
    - Boosting
        - 모델이 학습 + 예측 -> 잘 못맞춘 데이터 생성
        - 잘 못맞춘 데이터를 더 잘 맞추기 위해 노력하는 모델 생성
        => 위 과정 반복
- Popular Clustering
    - K-means
        - centroid를 이용해 주변 membership의 cluster 업데이트
        - cluster를 이용해 centroid 갱신
        => 더 이상 cluster가 안바뀔 때 까지 갱신
    - Mixture of gaussian
        - K-means의 일반화 버전
        - 각 그룹에 속할 확률을 보여줌