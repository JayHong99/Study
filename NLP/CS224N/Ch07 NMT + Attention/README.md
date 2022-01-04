# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# NMT + Attention (2021. 12.26)

## NMT
### Pre-NMT
- 인간지능
- SMT
    - 통계 기반의 Machine Translation
    - "source language 문장을 target language 문장으로 번역했을 때, 가장 가능성이 높은 문장은?"
    - alignment를 잡아내 번역
        - 어떤 단어가 어떤 단어와 연관이 있는지 매칭
        - 모든 단어가 매칭되지 않는 등 다양한 문제 존재
    - SMT로 문장 Decoding
        - 문장을 풀어헤쳤을 때, 주어진 단어를 모두 사용해 문장을 헤칠 수 있는 방법을 학습
    - 시간, 사람, 돈이 많이 필요
### NMT
- End-to-End with 2 RNNs
    - Encoder - Decoder
    - Encoder : 입력 문장이 들어오는 RNN
    - Decoder : Encoder에서 반환되는 State를 해석해 Decode하는 RNN
- NMT 학습
    - 앞선 단어가 주어졌을 때, 다음 단어가 나올 확률에 대한 Backprop
- Multi-layer NMT
    - 2017 Paper : Encoder는 2~4, Decoder는 4층 구조가 제일 좋음
    - 층수가 높을 수록 복잡한 관계 해석(Sentiment Analysis 등)
    - 층수가 낮을 수록 단순한 관계 해석(NER, POS 등)
    - stacked RNN이라고도 부름
- Beam Search
    - P(w_1) > P(w_2) 라고 해도, P(w_1)oP(w_3) < P(w_2)oP(w_4)일 수 있음
    - 가장 높은 확률로 하나만 탐색한다고 해서, 그 문장이 가장 높은 확률인 것은 아님
    - 모든 문장 탐색 => Computing 자원의 낭비
    - k개의 가장 가능성이 높은 문장들만 남기면서 decoding 실행
        - 특정 문장 길이 T를 넘어가면 stop
        - 완성된 문장 n개를 만들어보고, 가장 확률이 높은 문장 선택
- NMT 특징
    - 장점 : 성능적인 측면과 인간의 노력이 덜 필요
    - 단점 : 해석과 통제가 어려움
    - 성능 측정
        - BLEU Score : 인간 번역과 기계 번역 사이의 유사도 검정 (항상 좋은건 아님)
    - 한계점
        - OOV, Domain Mismatch, 대화 맥락 파악, 데이터 부족, 은유법 등 이해 부족, 서로 다른 문법적 구조, 문맥상 일맥상통을 파악하지 못하는 등의 문제가 있음

### Attention
- 기존 Bottleneck Problem 해결
    - Bottleneck : 병목 현상 : Encoder이 output에 너무 많은걸 담으려니 생기는 문제
- 방법
    - Decoding의 각 step에서 Encoder의 hidden state와 점곱을 통해, 가장 유사한 단어를 찾아 softmax 적용
    - sotmax를 통해 나온 단어를 바탕으로 다음 step을 진행, EOS가 나오면 중단
- 특징
    - NMT 성능 향상
    - 병목현상, Vanishing Gradient 해결
    - 해석의 여지