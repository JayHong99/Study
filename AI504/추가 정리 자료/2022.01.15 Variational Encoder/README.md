# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

---

**스터디 목표** : 

Variational Encoder에 대해서 추가적으로 정리했습니다.
https://jayhong1999.notion.site/More-about-Variational-Encoder-bb6cc04204a04e1f91a5dc4d3f577194


        <Variational Encoder의 시작>

        [기본 Auto Encoder의 문제점] 
        => Generation 단계에서, Auto Encoder는
        Input -> Hidden DIm -> Output
        이 3단계를 거쳐서 진행하기 때문에,
        Hidden Dim에 대한 통제가 어려움

        [해결법]
        Latent Space가 I -> H -> O 로 가는 과정에서
        기존 : Input의 정보를 압축한 저장공간
        해결 : Input의 데이터의 분포 저장공간

        즉, 아래의 과정에서 이점이 있음
        기존 : 완전한 데이터를 넣어 다 돌려야함
        해결 : 데이터의 분포를 넣어서 Decoding만 사용

        [VAE에 대한 직관적 이해]
        - 연속성 : 비슷한 두 점은 Decode 했을 때, 비슷한 결과를 초래해야 함 
        => A라는 데이터와 B라는 데이터 사이의 C라는 데이터를 이용해서 Decode했을 때, A와 B의 특징을 반반 가지고 있어야 함
        - 완전성 : 그 결과가 의미 있어야 함

        => 이 두가지는 Latent Space를 "분포"로 접근해서 해결할 수 있다.

        * 이제부터 "Latent Space"는 H의 "분포"를 의미함

        <Variational Encoder의 구조>

        [AE 구조 vs VAE 구조]
        AE : Encoder -> latent representation -> reconstruction
        VAE : Encoder -> latent distribution -> sampled latent representation -> reconstruction

        [특징]
        1.  Regularisation(과적합 방지) : Latent Space에 너무 많은 정보가 담길 수 있음 
        -> 그러면 Generation의 효과도 떨어지고 의미 없어짐

        2. 데이터의 분포를 학습
        Latent Distribution은 표준 가우시안 분포를 다른다고 가정함.
        => 공분산행렬과, 분포의 평균을 받을 수 있음(표준 가우시안 분포의 특징)
        => 분산과 평균을 조정하며 정규화하기 좋음(regularisation)
        => Latent Distribution 속 임의의 랜덤 z를 뽑았을 때, 표준 분포 위에 있기 때문에, 좋은 z를 뽑을 확률이 높아짐 (표준 가우시안 분포 + regularisation의 특징)
        => Generation에 있어서 효율 Up (Reconstruction Error를 통해 측정)

        [Loss Function]
        2가지 Loss가 필요함
        1. Regularisation Loss
        => 표준 가우시안 분포를 따른다는 Loss
        => Latent Space가 표준 가우시안 분포를 따른다는 계산이 어렵기 때문에, KL-divergence를 이용해 근사치 (추정치)로 계산

        2. Reconstruction Loss
        => Encoding - Decoding의 관점에서, 데이터가 주어졌을 때, 복원을 잘 하는지에 대한 측정치

        근데, 2 가지 Loss가 서로 Trade-off 관계임
        즉, 하나가 올라가면 하나가 내려가는 관계
        => 이 두가지 Loss를 사용하여 적절한 선을 학습하는게 목표

        <정리>
        Variational Encoder는 Auto Encoder의 Generation의 문제를 해결하기 위해,
        1.  Latent Space를 표준 Gaussian 분포를 따른다고 가정해 Regularisation Loss를 최소화
        => 표준 Gaussian 분포를 따르면, 공분산행렬과 평균을 알 수 있는데, 이를 이용해서 정규화를 할 수 있음
        => 즉, 공분산 행렬과 평균을 얻어내고, 잘 조정해서 데이터의 분포를 제어함
        => 이를 통해 과적합 방지등의 이점이 있음
        => 표준 Gaussian 분포를 직접 추정하기란 계산 비용이 막대하기 때문에, KL-Divergence라는걸 이용해서 근사치(추정값)을 이용
        따라서, 이 값을 조정하는 Loss 하나가 필요

        2. 데이터의 분포를 통해 추출한 Random 값 z를 Decoder에 넣었을 때의 Reconstruction Error를 최소화
        => 해당 Encoder의 목적 자체가 Reconstruction에 있기 때문에, 얼마나 잘 복원하는지 측정
        이 값을 조정하는 Loss 하나가 필요

        3. 두 Loss 제어
        => Regularisation Loss와 Reconstruction Loss 는 서로 Trade-off 관계임
        => Gradient Desecent를 이용해서 학습이 가능
        => 역전파 알고리즘을 통해 적절한 균형을 찾아나가면 됨