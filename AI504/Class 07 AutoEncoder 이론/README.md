# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 7 Autoencoder 이론
- More Study Info at : https://jayhong1999.notion.site/Class-7-AutoEncoder-1d44988706344ad0acefd27c7139b336

## Latent Representation
### 뜻
- 숨겨져 있는 표현
- DL 에서는 Hidden Layer의 값이라고 생각
- 1 : 1의 관계보단 M : N 의 관계라고 판단

### 학습
- Cross Entropy를 Minimize 하도록 학습
- Non-Linear Activation을 거치면, Latent Representation을 통해 Output을 더 잘 맞출 수 있도록 차원을 왜곡하는 등을 수행

## Autoencoders
### 기본 구성
- Encoder
    - X ->  Z
    - Input -> Latent Space
- Decoder
    - Z -> X'
- Loss
    - MSE
    - 자기 자신과 최대한 유사해 지기 위해서 복원

### Multiple Layers Auto Encoders
- X -> h -> Z -> h' -> X'
    - layer가 많아질 수록, 왜곡을 더 많이 할 수 있음
    - 그러면 목표를 더 잘 찾아갈 수 있음
    - example)
        - NLP에서 단어 15만개로 topic 분류한다고 하자.
        - Auto Encoding으로 15만 단어 -> 4차원 -> 15만 단어라고 하면
        - 중간 단계가 많이 생략되어있다보니, 잘 못찾아낼 가능성이 큼

## Autoencoder VS PCA
### PCA
- Reconstruct Error를 최소화 하는데 목적 == 분산량을 최대화
- 선형 변환
- K개의 변수를 선택 -> variance가 작은 값들은 제거
- 고차원으로 mapping하기 위해서는 kernel을 사용해야 함

### Autoencoder
- Reconstruct Error를 최소화 하는데 목적 == 분산량을 최대화
- 비선형 변환
- K Dim으로 압축함 -> 암묵적으로 작은 분산을 무시하게 됨
- 고차원으로 mapping하기 쉬움

## Training Process = 기본적인 학습 과정
### 전체 과정
- 데이터 split -> 데이터 가져오기 -> 모델, loss 정의 -> Optimizer 정의 -> 반복 학습 -> 평가

### Data Split
- train / validation / test로 쪼개기
- 8:1:1 , 6 : 2 : 2 등 실제에 맞게 사용
- N-fold 도 고려할 것
- sklearn train_test_split

### Load Data
- 모든 데이터를 memory에 넣으려면 너무 무거움
- disk에서 불러오고 나가고를 반복하자
- torch.utils.data.Dataset or torch.utils.data.DataLoader

### 모델, Loss 정의
- 모델 정의 :  torch.nn.Module
- Loss 정의 : torch.nn.* or 알아서 만들기

### Define Optimizer
- SGD, Adagrad, Adam ...
- 굳이 새로운 optim은 만들려하지 말자
- use torch.optim.*

### Training Loop
- N Epoch 동안,
    - K batch 만큼 반복하면서,
        - random mini-batch (X,y)를 가져와
        - X를 모델에 넣어 예측값 y’ 을 얻어냄
        - Loss(y,y’)를 계산해서
        - Optimizer로 모델 M의 파라미터를 업데이트 함
    - Validation Set으로 평가
        - validation에서 좋은 성능을 가지면 모델 M을 저장
        - Fetch next minibatch (X,y)
        - Push X through M
        - Save batch performance
        - Calculate Validation Performance
- 최고 성능 모델 M 을 가져와서
    - 최고 파라미터를 불러오고
    - test set을 T batch로 반복해서 예측하고
    - 성능 평가

## 시각화
### 개념
- 이미지 1과 1 쌍은 이미지 1과 5 쌍보다 가깝게 그려져야 함
- 이미지는 2D이기 때문에, 평면에 매칭하기 어려움
- => 데이터를 압축해야 함

### PCA
- 분산을 기반으로 함

### t-SNE
- L2 Norm 거리 기반
- 자신과 가까운 애들 사이의 거리를 측정해서 매핑

### UMAP
- L2 Norm 거리 기반
- t-SNE 보다 더 효과적인 매핑 및 한계 극복

## Autoencoder Variants
### Sparse Auto Encoder
고차원 Mapping
- Sparse한 Latent를 유발한다.
- 기본 Loss에 Sparsity를 유발하는 오메가를 더해 sparsity를 유발
    - 방법 1 : KL-divergence 사용
    - 방법 2 : L1 regularizer 사용

### Denoising Autoencoder
내용
- X에 Random Noise를 추가한 후에, 원본 X 복원
- 작은 NOise를 무시하면서, 중요 패턴 위주로 복원
    - PCA와 비슷하네

학습
- X -> noise -> X tilde -> encoder -> Z -> decoder -> X'
- Loss Function은 X로 X'을 예측하는 것임
- noise는 주로 Gaussian noise를 줌