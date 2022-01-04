# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# Dependency Parsing + RNN(2021. 12.21)

## Dependency Parser
- 기존의 문제점
    - Sparse, 미완성의 문장, 계산 비용
- 1차 해결 : Dependency Embedding 적용해 SVM
    - Dependency Embedding에 POS가 내포되어 있었음
- 2차 해결 : Softmax Classifier
    - Non-linear한 NN Classifier를 이용하니, 성능 향상
- 성능 추가 향상 : Graph-NN
    - 그래프 기반으로 모델링하니 성능 향상
    - 다만, 추론 시간이 너무 오래 걸림
        - 문장 길이가 길면 더 오래걸리지

## NN tips
- Regularization : 과적합 방지
- Dropout : 과적합 방지
- Vectorization : 벡터화
- Non-linearities, old and new : 과거부터 현재까지 Activation Function
    - Adam 기억
- Paramter Initialization
    - 파라미터 초기화 방법
    - 과적합 방지
    - Xavier initialization 기억
- Optimizers
    - SGD, Adagrad, RMSprop,Adam, ...
    - Adam이 좋음
- learning rate
    - 학습률

## RNN
Q) "the Students opened their ???". 여기서 "???" 에 들어갈 단어는?

기존의 방법
Onehot Input -> Embedding -> NN Classifier
the 가 나왔을 때 Student가 나올 확률 -> the, Student가 나왔을 때 opened 가 나올 확률의 과정을 거쳐 모델링 결과를 예측

이 방법의 문제점
- 필요 없는 단어가 많이 들어감

Idea 1
마르코프의 가정
- 해당 단어를 이루는데 영향을 끼치는 단어는 앞선 n개이다.
- t+1번째 단어를 예측하는데는, n-1gram만 보면 된다.
- 

Idea 2 
Solving Sparisity Problem
등장하지 않았던 패턴에 대한 방법

=> RNN
앞에서 나온 단어들을 hidden state로 넣어 해당 결과를 모델에 집어넣는 과정
인간의 사고방식과 비슷
장점
- 어떤 문장 길이든 사용 가능
- 모델의 크기가 더 커지지는 않음
단점
- 시간이 오래 걸림
- 예측 단어와 멀리 떨어진 단어의 hidden state는 정보가 많이 손실됨
