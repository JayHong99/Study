# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 9 Variational Encoder
- More Study Info at : https://jayhong1999.notion.site/Class-9-Variational-Autoencoder-27a6746b82264db9ba8dce1d44697d51

## Generating Samples
- 새로운 데이터를 어떻게 만들어낼까?
- Perturbing z
    - Latent Space에 있는 값들을 이용해서 label의 space 속 값을 이용해 Generation 할 수 있지 않을까?
    - 그럼에도, 완전히 새롭거나 다양한 sample을 만들지는 못함

## Vairational Inference
### Posterior Distribution (사후 확률 분포)
- 주어진 데이터 X를 이용해 Z를 추정하고, 이를 이용해 X'을 추정해야 한다.
    - 즉, X를 결정하는 Z를 추정해야 한다.
    - 이 과정에서 latent는 p(z)를 따르고, x'은 p(x|z)를 따름
- Z에 대해서 정확히 추정하자.
    - X에 대한 Z의 참된 사후확률 분포식은 다루기 어려움
    - Bayesian Rule을 적용하면 추정할 수 있음
    - P(Z|X)를 Bayeisan Rule을 이용해서 해석하고, P(X)를 상수의 형태로 생각하면, P(Z|X)는 우도와 Z와 유사한 분포 Q(Z)를 띄게 됨
- Q(Z) 학습
    - 최종적으로 P(Z|X)를 알기 위해서는 Q(Z)를 학습해야 한다.
    - 이는 KL-Divergence를 따라 해석 가능하다.
    - KL(Q||P)를 추정할 것인데, 보통은 KL(P||Q) 를 추정한다.
        - KL의 왼쪽에는 주어진 값을, 오른쪽에는 추정할 값을 둠
        - KL(Q || P) 는 원래와 반대이므로 ,Reverse KL이라고 부름
    - Original KL(P||Q)는, Q에 대해서 P의 Divergence를 학습하므로, Q가 존재할 수 있는 모든 공간에 대해서 z를 추정한다.
    - 하지만, Reverse KL(Q||P)는 P에 대해서 Q의 Divergence를 학습하므로, P가 존재할 수 있는 공간 근처에 Gaussian 분포 만큼의 거리를 둔 채로 추정한다 => Reconstruct 할 때, 더 유사하게 복원할 수 있다.
    - 학습 방법
        - KL을 logP(x)에 대해서 해석하면, ELBO와 KL로 풀 수 있다.
        - ELBO와 KL은 Trade-off 관계로, ELBO를 최대화하면, KL을 최소화할 수 있다.
        - 따라서, ELBO를 최대화 하도록 학습한다.
## Variation Autoencoder
### Loss
- Reconstruction Loss와 Reconstruction Term을 이용한다.
- Q(Z)에 대해서 lambda를 추가하고, lambda는 평균과 표준편차에 대한 정보를 담고 있다.
- 즉, KL을 다룰 때, 원본의 분포 근처의 Gaussian Distribution을 적용하기 위한 평균과 표준편차를 학습 단계에서 함께 배워, 가장 적합한 분포를 선정한다.
### 학습 과정
- Sample Data x를 Encoder에 넣어 평균과 표준편차를 얻는다
- Gaussian 정규분포를 다루는 z를 추정한다.
- z를 decoder에 넣어 x'을 추정한다.
- MSE와 KL을 계산한다.
- 역전파를 통해 파라미터를 업데이트한다.
