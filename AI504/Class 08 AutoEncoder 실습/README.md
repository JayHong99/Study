# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 8 Autoencoder 실습
- More Study Info at : https://jayhong1999.notion.site/Class-8-AutoEncoder-9effc5299d044812bcfdc32668f11162

## 실습 Data

- Mnist Data
- Train : valid : test = 5 : 1 : 1

## Model

### Normal Autoencoder
- Encoder
    - Input (784) → Hidden(100) → Hidden(32)
- Decoder
    - Hidden(32) → Hidden(100) → Output(784)
- Loss
    - MSELoss()
    
- Optimizer
    - Adam
    
⇒ 기존 이미지에서 특징을 잡아냄

### t-SNE
- 결과의 분포를 살펴봄
- 학습이 덜 되면, 잘못된 분포를 확인함


### Denoising Autoencoder
- Encoder
    - Input + Noise → Hidden → Hidden
- Decoder
    - Hidden → Hidden → Output
- noise
    - nn.init.normal_(~~)
    - 항상 Noise를 추가하는 건 아님
    - MNIST는 단순한 이미지이기에 괜찮음
    - 일반적인 이미지에 잘못 Noise를 넣으면 사람이 사라지는 등 문제가 생길 수 있음