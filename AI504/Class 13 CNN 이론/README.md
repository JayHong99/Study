# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 11 Gnerative Adversarial Networks 이론
- More Study Info at : https://jayhong1999.notion.site/AI504-Study-3254606f482e4b7c893a82189275c941

## CNN
### 생물학적 Motivation
- 인간은 점 -> 선 -> 면 -> 객체 순으로 인식하고 판단함
- Model도 이처럼 특징들을 합쳐나가면 인식에 도움이 될 것이라 생각

### History
- LeNet-5 : (Conv -> Sampling) * M -> FC : 최초의 CNN
- AlexNet : 5 Conv Layer : 깊게 쌓기의 시작
- Modern ConvNet : 1D,2D,3D 모두에서 분류, 검색, Detection, Segmentation 등 사용

### Fully Connected Layer
- 32x32x3 image를 1차원에 3072 x 1로 바꾸어 학습하면, 공간의 정보가 없어짐

### Convolution Layer
- Filter : 필터
    - 필터의 값 만큼, 픽셀 단위로 훑으면서 판단함
    - ex) 7x7x3 image에 3x3x3 filter를 적용하면, 우상단부타 좌하단까지 3x3x3 필터를 이용해서 한칸씩 이동하며 특징을 잡아냄
    - 개별 필터에 대해서 Weight Matrix와 Bias를 학습함
    - Filter를 여러개 사용하는게 보통이고, Filter의 수가 Output DImmension이 됨
    - 개별 필터는 같은 데이터도 조금씩 다르게 판단함
- Receptive Field
    - 3x3 필터 3 layer는 7x7 필터 1 layer와 같은 역할 수행
    - 하지만 파라미터는 기존 3x3x3 = 27 대비 7x7x1 = 49로, 훨씬 효율적
    - layer가 깊어질 수록, 좀 더 자세한 특징들을 잡아가며 학습 진행
- Stride
    - 몇 pixel씩 건너뛰며 볼 것인지 정의
    - 7x7x3에 3x3x3 필터를 stride 2로 지정하면,
        - 1~3 조사 필터, 3~5 조사 필터, 5~7 조사필터 3개로 이루어짐
    - stride를 적용했을 때, 데이터가 사라지진 않는지 유의하자
        - 즉, (N-F) / Stride + 1의 값이 정수가 아니면 유의하자
- Padding
    - 이미지의 공간 정보 형태를 그대로 유지하고 싶을 때, 바깥은 0으로 채움
        - 기존 : 7x7x3 image + 3x3x3 Filter -> 5x5x1 Image
        - 패딩 : 7x7x3 image + 3x3x3 Filter -> 9x9x3 image로 변경 후 적용 -> 7x7x3 image
- 1x1 Convolution Layer
    - Channel의 관점에서 정보 압축
    - Fiber라고 부름
    - example) 56x56x64 + 1x1x32 -> 56x56x32
- Pooling
    - Downsampling의 관점
    - MaxPooling은 필터에서 가장 큰 값을 골라냄
    - AveragePooling은 필터의 평균을 적용함
- 1D ConvNet : 1차원 데이터에 적용 (문장, 오디오, 시계열 등)
- 3D ConvNet : 3차원 데이터에 적용 (CT, Video 등)

### Training
- Dropout : p %의 hidden node를 0으로 만들어 연결을 끊는 방법
    - Regularization의 방법 중 하나
    - Network 자체의 power를 낮춰버리는 역할
- Normalization
    - N(0,1)을 따르게 만듬
- Distribution Shift
    - 학습한 데이터의 분포와 Test 할 데이터의 분포가 다른 현상
    - ML에서 자주 등장하는 문제
- Internal Distribution Shift
    - Layer의 Output은 다음 Layer의 Input이 됨
    - 만약 Output의 분포가 서로 다르다면, 학습하기 어려움
    - 내부적 Distribution Shift 발생
- Batch Normalization
    - Batch 단위로 Normalization 적용
    - Scale하고, gamma, beta를 이용해 원본 데이터의 특성을 좀 내줌
    - 따라서, test 예측할 때도 어려움이 있음
    - NxCxHxW 형태를 1xCx1x1 형태로 바꿔버림
- Layer Normalization
    - 모든 데이터에 대해서 Normalization 적용
    - train, test 동일하게 사용하고
    - RNN 에서도 사용 가능
    - N x 1 x 1 x 로 바뀜
- Instance Normalization
    - 공간 정보를 압축함
    - N x C x H x W -> N x C x 1 x 1 로 바꿈
    - Style Transfer 등에서 사용

### CNN Architectures
- ResNet
    - Skip Connection이라고도 부름
    - 현재의 정보를 다음 layer output에 보태는 형식
    - 다음 layer에서 얻어갈 정보가 없더라도, 정보 전달을 위해 없앨 수 없었음
    - ResNet을 통해, 의미가 없으면 W는 I로, b 는 0 으로 만들어버릴 수 있음

### Project
- CelebA 데이터를 이용해 1,000장의 test image의 FID 를 20 이하로 만들어라.
- 모델은 DCGAN 등 다양하게 찾아보고 적용해라.