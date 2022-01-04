# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 2 : Numpy 실습
- More Study Info at : https://jayhong1999.notion.site/AI504-Study-3254606f482e4b7c893a82189275c941

## 1) Scalars, Vectors, Matricies
- Scalar : 0차원
- Vectors : 1차원
- Matrix : n차원
- 각각에 대한 정의와 shape 확인

## 2) Tensors(N-dimensional Array)
- N 차원 Array 정의, shape, ndim 확인
- 다양한 차원의 Array 정의 및 확인

## 3) Defining Numpy Arrays
- np.ones, np.zeros, np.full, np.random.random, np.arange로 Array 만들기
- 위의 값을 array.reshape()로 원하는 차원 형태로 변환

## 4) Indexing & Slicing
- Indexing과 Slicing Operation에 관해서 다양한 실습 진행
- Idx, Boolean 등을 이용해서 여러 차원 단위에서 원하는 Vector나 Scalar, Matrix를 추출하는 방법

## 5) Math Operations
- 동일한 dimmension을 가진 matrix에서의 사칙연산 + 통계치
- np.dot() 과 @ 연산자의 차이

## 6) Shape Manipulation
- 서로 다른 shape를 가진 matrix의 형태를 맞춰주기
- np.reshape(~~,-1) , None
- np.vstack, np.hstack, np.concatenate()
- np.transpose()

## 7) Broadcasting
- 서로 다른 차원을 가진 matrix or vector의 연산
- 차원이 다르면 차원을 맞춰서 연산해주는 numpy의 기능

## 8) Final Quiz
- Sigmoid 함수 구현