# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 6 PyTorch + Neural Network 실습
- More Study Info at : https://jayhong1999.notion.site/Class-5-Logistic-Regression-Neural-Network-289ae60a4e4c49d69969ec8db162de89

## Pytorch 개념
- 직관적이고 코드가 간결함
- Numpy와 유사
- 요즘 인기가 더 많아지고 있음

## AutoGrad
- 자동으로 Gradient를 계산해줌
- requires_grad를 통해
- retain_grad : 저장된 gradient를 불러옴
- torch.no_grad() : gradient 학습 없이 사용 (예측 단계 등)

## nn.Module
- NN Linear
    - Input->Output
- Optimizers   
    - SGD : Stochastic Gradient Descent, 상세 생략
- MLP
    - Hidden Layer가 반영된 NN