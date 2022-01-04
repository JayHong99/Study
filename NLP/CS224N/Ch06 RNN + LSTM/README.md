# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# RNN + LSTM (2021. 12.23)

## RNN
### Loss
- RNN의 Loss = 다음 step의 값을 정확히 맞추었는가?
- RNN Loss Update : 
    - 1) batch 단위로 합쳐서 한번에 update
    - 2) step마다 backpropagation을 적용해서 매번 차감
    - 3) 1)과 2)의 절충안 : 20step정도마다 합쳐서 update

### Hidden State
- 지금까지 모든 step의 정보를 압축한 정보
- 이를 기반으로 예측하면 모든 정보를 가지고 예측한 것과 동일

### 평가
- Perplexity
    - 혼란도
    - 지금까지 이런 단어가 나왔을 때, 다음 단어가 나올 확률은? 에 대한 정보
    - 문장이 길어질 수록 이 값은 올라감

### 추가적 특징
- RNN은 Input Length에 무관하게 적용 가능 (어차피 Weight도 하나)
- Step마다 동일한 weight matrix
- NLP에서는 다음 단어를 예측하는데 사용

### RNN의 활용
- POS, NER : 각 hidden state마다 예측
- 감성분석 : 마지막 hidden State로 예측
- Encoder : 질문에 대한 답변
- Decoder : 음성 -> 글자

### RNN의 문제점
- Vanishing Gradient 
    - 미분 계수가 작으면 초반 단어의 영향력이 너무 낮아지는 문제
    - LSTM으로 조금이라도 해결
- Exploding Gradient 
    - 미분 계수가 크면 무한대로 커버림 
    - Gradient Clipping : 미분계수의 최대치 설정

## LSTM
### 내용 요약
- 입력 : Hidden State, Cell State, Input
- Action
    - Forget Gate (Erase)
        - 이전까지의 Hidden State와 현재 Input으로 지울 정보 계산
        - cell t-1에 multiply => 지울 정보에는 작은 숫자
    - Input Gate (Read)
        - 이전까지의 Hidden State와 현재 Input으로 저장할 정보 계산
        - forget gate의 결과에 add => 현재 정보 추가
    - Ouput Gate (Write)
        - 이전까지의 hidden state와 현재 Input으로 내보낼 정보 계산
        - 현재 hidden state = Input gate 결과에 tanh를 한 값의 곱
        - 현재 hidden state는 별도 처리 없이 내보냄

### Vanishing Gradient 조금 해결
    - Forget, Input, Output Gate 모두 sigmoid 함수를 통과
        - 그 값이 매우 작아짐
    - 각 gate는 모두 각자의 Weight가 존재 
    - 남기고 싶으면 1에 가깝게, 지우고 싶으면 0에 가깝게 하면 됨
        => 이러면 초반에 있는 값도 끝까지 살아있을 수 있음
    - Direct Connection
        - 값을 직접 연결해 해결하려는 노력
        - ResNet

### Bidirectional LSTM
"I am lazy man"
- LSTM으로는 "am"을 가지고 "lazy"를 예측해야 함
    => man을 가지고 하는게 좀 더 효율적일 것임
    => 쌍방향으로 분석하자
- 정방향 Weight와 역방향 Weight를 모두 계산해 나오는 hidden state를 만들어 그 값을 현재 step의 hidden state로 만듦

추가 질문
Q) 언제 RNN을 사용하고, 언제 LSTM을 사용하냐?
A) 그냥 LSTM을 사용해라. 연구로써 Simple RNN 사용하는거 아니면 그냥 LSTM이 좋음