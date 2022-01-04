# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 4 : Basic Machine Learning 실습
- More Study Info at : https://jayhong1999.notion.site/AI504-Study-3254606f482e4b7c893a82189275c941

## Packages
- Numpy -> Class 2
- matplotlib.pyplot as plt -> Plotting
- sklearn -> The most famous Machine Learning Packages

## Data Plotting
- plt.plot(x좌표, y좌표) => 선
- plt.scatter(x좌표, y좌표) => dot plot

## Generating Samples
- 기본 데이터에 noise 더하기
    - np.random.unifrom(mu, sigma, size)를 통해 Gaussian noise 추가 가능
- sklearn.preprocessing.Polynomial Features
    - exapnd dimenssion of features
    - degree = 2가 default
    - 자기 자신의 제곱 등 다양한 Feature 생성

## Regression (Overfitting, Underfitting)
- sklearn.linear_model.LinearRegression
    - 다중선형회귀모형
    - MSE로 학습이 Default
- sklearn.linear_model.Ridge
    - L2 Regularizer가 반영된 선형회귀모형

## Data Loading
- Iris data : sklearn.dataset에서 불러옴
- sklearn.model_selection.train_test_split
    - train, test data 쪼개기

## Classification
- sklearn.linear_model.LogisticRegression
    - Logistic 회귀분석 모델

- sklearn.svm.SVC
    - Support Vector Machine의 Classifier Model

- sklearn.tree.DecisionTreeClassifier
    - 의사결정 나무
    - max_depth : 나무의 깊이 -> 과적합 등 조절
    - Non-linear Classification이 특징
        - 선형회귀로 분류하기 어려운 문제 해결 가능


- 고차원 데이터를 PCA를 통해 저차원으로 매핑할 수 있음
