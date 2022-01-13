# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 11 Gnerative Adversarial Networks 이론
- More Study Info at : https://jayhong1999.notion.site/AI504-Study-3254606f482e4b7c893a82189275c941

## 복습
### VAE
- X는 P(Z|X)를 따라 z로 압축하고
- z는 다시 X'으로 재구축하는게 Auto Encoder 모형
- 여기서, z를 X'으로 재구축 할 때, z가 X'의 모양을 비슷하게 따라간다고 가정한 게 VAE
- 즉, True P(Z|X)를 이용해 P(X|Z)를 구하는 방법
- 이 과정을 거치면, Q(Z)가 원래 범주와 유사하게 되어, Reconstruction 했을 때의 효과가 좋아짐

### Generative Models
- Pixel - CNN
    - 픽셀 단위로 예측
    - 앞에서 이러한 값들이 주어졌을 때, 현재 픽셀의 값은? 의 개념
- Wave Net
    - 음성의 1 프레임 단위로 예측
- GPT-3
    - 단어를 1 토큰 단위로 예측

## GAN

### 개념
- 생성망 G와 판별망 D가 서로를 속고 속이는 관계
- G는 D를 속이는게 목적이고, D는 G에게 안속는게 목적 -> Binary Classification

### 학습
- Binary Classification in GAN
    - D의 관점에서, D의 파라미터를 수정하며 Real, Fake를 잘 분류하자
    - G의 관점에서, G는 Latent Rrepresentatio인 Z를 이용해 D를 속이자
        - 이 때, G는 D가 Real을 분류하는데는 관심이 없고, Fake를 분류해내지 못하도록 하면 됨
        - 즉, G의 파라미터를 학습을 통해 Z가 더 잘 만들어지도록 학습

- 과거 MinMax Game
    - G가 이기는 것과 D가 이기는 것 사이의 trade-off 관계를 이용해 학습
    - 최종적으로 D와 G가 50%의 균형을 이루는 "Nash Equilibrium"이 목적
    - 하지만 이 관점에서, 초기 G가 잘못 만들어질 경우 학습이 안되는 문제 발생
        - G가 너무 못만들어짐 -> Fake를 모두 분류 -> Loss가 0이 됨 -> 학습 불가능 상태
- 수정된 Min Max Game
    - G를 Maximize하도록 학습하자!
    - 수식으론 다르지만, 내용은 동일하고, Nash 균형을 사용하지는 않는 상태

### GAN 학습
- 전체 epoch동안
    - 판별기 D 학습
        - Random Noise로 만들어진 X'과 실제를 분류하는 작업을 통해 D의 Loss 계산
    - 생성망 G 학습
        - 생성망 G는 Fake를 잘 분류해낼 수 있는지 Loss 계산
- 학습 중지
    - 절대적 규칙은 없음
    - 몇 epoch마다 이미지를 출력해서, 잘 되면 Stop
    - FID를 이용해서 Stop
    - GAN 변형을 통해 Loss를 보고 Stop
    - 기타 등등 다양한 변형이 존재

### Mode Collapse
- G가 매우 현실적인 Fake 하나 (혹은 몇개)를 만들어냄
- 이를 D가 판별하지 못함
- 그러면 D는 현재 state가 잘 되고 있다고 판별
- Local Minima에 빠짐
- 해결법은 Tuning과 GAN 변형 등이 있음

## VAE vs Autoregressive vs GAN

### VAE
- 0~1의 normal distribution을 이용해 생성
- x와 x' 사이의 loss를 통해 학습
- 흐릿한 이미지가 자주 나옴

### Pixel - CNN
- 시작 픽셀 몇개를 주고, 나머지 픽셀을 생성
- Sample의 우도를 사용해 학습
- VAE보다 나은 성능을 보임

### GAN
- Uniform or Gaussian Distribution을 이용한 데이터를 Generator에서 생성
- 성능은 FID 등 기타 여러가지로 평가
- 최근에는 1024x1024 이미지도 선명하게 만들어내지만, Mode Collapse 문제있음

## 평가지표

### Inception Score
- Inception V3라는 Pre-trained Model을 사용
- 2가지 평가지표
    - Image Quality : 이미지가 깨끗한가?
        - z를 이용해 생성한 이미지가 V3에 넣었을 때, 출력되는 확률이 한 쪽에 쏠려있다면 , 잘 만든 이미지라고 판별
    - Image Diversity : 이미지가 다양한가?
        - 예측한 이미지가 여러가지 (즉, entropy가 높다)라면 다양하다고 판별
- 2가지 평가지표를 KL-divergence를 이용해 평가
    - KL(p(y|x) || p(x)) 를 평가지표로 사용
        - KL(P||Q) = -sum(plog(q/p))
- 최하점은 1, 최고점은 N
    - 모두 균등하게 분포시켜 거리가 1이면 최하
    - 모두 다양하고 고루 나오면, N개의 Label이 최고

### FID Score
- V3 모형에서 Pooling 한 Vector Distribution을 사용
- X와 Z를 통해 만든 X'을 V3에 넣고, 출력되는 값이 얼마나 다른지를 판별
- 이 거리가 짧다면 두 이미지가 유사함을, 길다면 많이 다름을 의미
- 두 가우시안 사이의 Frechet 거리 계산

## Applications of GAN
### Text-to-Image GAN
- 조건부 이미지 생성
- Text Encoding을 Generator와 판별기에 넣어서 봄

### Cycle GAN
- 스타일을 바꾸는 모형
- 적대적 loss와 순환적 일관성

### Style GAN
- 고해상도 이미지에 사용
- z를 MLP에도 적용하는 등 다양한 기술 접목