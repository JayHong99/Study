# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# Self-Attention + Transformers (2021. 12.26)

## RNN + Attention
### RNN for NLP
- 기존 : Encoder에 Bi-LSTM Decoder에 LSTM + Attention으로 NLP에 적용하곤 했음
- 문제 : Encoder 내부에서 수식어나 문장어구가 많아지면 time step의 의미가 있는 RNN은 어려워짐
- 해결 : Self-Attention

## Self-Attention
### 구성
- Query, Key, Value로 구성해 살펴봄
- 3가지 모두 X에서 파생되어 나와 동일한 값을 지님
- Input Matrix X를 가지고 XQKX를 하면, Query 단어와 Key 단어 사이의 유사도를 살펴볼 수 있음
- 이 값을 softmax를 통과시키면, Query단어와 Key 단어 사이의 유사도를 확률로 볼 수 있음
- 이 값에 XV를 더하면 각 단어의 값에 대한 가중치를 구할 수 있음

### 한계점 + 한계 돌파
- Attention은 문장 순서에 의미가 없다
    - Sequence Order
    - 문장 순서를 의미하는 pi 파라미터를 더하자
    - 이를 Positional Encoding or Position Representation이라 부름
- self-Attention을 몇층을 쌓아도 그냥 통과하면 weighted average와 동일
    - Nonlinearities
    - Feed Forward Network를 통과시켜 비선형성 추가
- Decoder에서는 미래 단어를 학습 단계에서 사용하면 안됨 (미래에 만들어질 단어를 살펴보게 됨)
    - Masking
    - 미래에 올 단어는 학습이 안되도록 -무한대 값을 부여

## Transformer
위에서 다룬 Self-Attention을 적용한 Transformer의 구성은 어떨까

### 추가 적용
- Key, Query, Value가 모두 동일한 값을 가진다
    - Key-Query-Value Attention
    - 모두 하는 역할이 다르기에 차이를 두자
    - 3가지 모두 개별 파라미터로 계산
- i번째 단어와 j번째 단어의 유사도가 높은데, 그 원인을 다르게 보고싶다 (일반화)
    - Multi Head Attention
    - Query단계에서 머리를 여러개로 만들어 다양한 관점에서 데이터를 살펴봄
- 학습 속도가 너무 느림
    - Residual Connection
        - 데이터를 건너 뛰어가며 전달
        - loss를 찾는 과정이 smooth 해짐
    - Layer Normalization
        - 의미없는 변동성을 제거
        - 값을 줄여 학습 속도 향상
    - Scaled Dot Product
        - Dot Product의 값이 너무 커지는 현상 방지
        - 좀 더 일반화된 Softmax 결과 획득
- Multi-Head Cross-Attention
    - 기존 RNN에서도 Decoder는 Decoding State를 Encoder State와 Attention 함
    - 이 과정에서 Encoder의 KV를 가져와 Decoder에서도 적용
    - Multi-Head가 적용된 상태에서 Encoder와 Decoder 사이의 Cross로 Attention을 적용한다는 의미

### Encoder Attention Block
- Input : Word Embedding + Position Representations
- Step1 : Input Multi-Head Attention
- Step2 : Step1 Layer Norm + Input Residual Connection 
- Step3 : Step2 Feed-Forward Network
- Step4 : Step3 Layer Norm + Step2 Residual Connection
- Other Attention Block or Decoder 

### Decoder Attention Block
- Input : Word Embedding + Position Representations
- Step1 : Input Masked Multi-Head Self- Attention
- Step2 : Step1 Layer Norm + Input Residual Connection
- Step3 : Step2 Multi-Head Cross-Attention
- Step4 : Step3 Layer Norm + Step2 Residual Connection
- Step5 : Step4 Feed-Forward Network
- Step6 : Step5 Layer Norm + Step4 Residual Connection
- Other Attention Block or Prediction

### 성능
- NMT에서 학습 파라미터 대비 훨씬 높은 성능
- Generation에서도 더 나은 성능

### 개선사항
- 문장 길이가 T면, Transformer는 계속해서 모든 관계를 보기 위해 T^2만큼 살핌
- 계산이 너무 오래 걸림
- Linformer : KQV에서 Q는 Projection 없이 바로 적용
- BigBird : Attention을 모두 다 적용하지 않고 Random + Window + Global 적용