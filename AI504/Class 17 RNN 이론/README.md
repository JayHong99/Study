# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 17 RNN 이론
- More Study Info at : https://jayhong1999.notion.site/AI504-Study-3254606f482e4b7c893a82189275c941

## RNN 등장 배경

기존 MLP, CNN의 문제
- 문장의 길이를 fix해야 함

Bag of Words로 해결 시도
- 모든 가능한 단어들을 Vocabulary로 만들고, 문장속에서 나타난 횟수를 Count함
- 이 경우에는 Dimension이 너무 커지는 문제
- RNN분야에서는 나타난 단어 1개만 1이고 나머지는 0이 되어버림
- 동일한 Bag-of-words이더라도 다른 의미를 지닐 수 있지만, 구별하지 못함
    - 나는 너에게 밥을 주었다.
    - 너는 나에게 밥을 주었다.

## RNN 구조

- time-step을 만들어 단어가 순서대로 들어가도록 구성

- x에 대한 weight matrix와
- h에 대한 hidden matrix를 하나씩만 구성

- 즉, 단어가 몇개가 들어가도 동일한 weight matrix가 적용됨
- => 이를 통해 기존 문장 길이 제한 해소

## RNN 적용 분야
### Sequence-level Classification
- 감성분석
    - 유치원 연극만큼 감명깊다. (Positive / Negative)
    - Notion에 RNN Applicaiton 법 정리
- Topic 분류 (multi-class classificaiton)
    - 스포츠, 정치, 시사, 경제, ...

### Classification / Regression at each step
- 언어 모형
    - "나는 아침에 []을 했다."에서 []에 들어갈 말은?
    - Notion에 RNN Applicaiton 법 정리
- POS
    - "나는 아침에 밥을 했다."에서 "나"의 품사는?

### Seq2Seq
- Translation : ex) 번역기
- Question Answering : ex) 챗봇

## Bidirectional RNN

기존 RNN의 한계
- 문장의 길이가 길어지면, 초반에 나온 단어의 의미가 사라짐
- Vanishing Gradient Problem

Idea
- 역순으로도 계산하자
- "I gave you my heart which you gave it away the very next day"
    - Forward RNN : I -> gave -> ... -> day
    - Backward RNN : I <- gave <- ... <- day
    - Forward의 day와 Backward의 I를 이용해 계산

## GRU
기존 RNN의 한계점을 마찬가지로 극복하기 위한 접근법

Idea
- Reset gate와 Update Gate 도입
- Update Gate 
    - x가 주어졌을 때, 이 정보를 얼마나 반영할지에 대한 gate
    - 0에 가까우면 이번에 주어진 x는 반영하지 않음.

- Reset Gate
    - 현재 step의 state를 만들 때, 
    - 이전 step의 state와 현재 step의 update gate를 거친 state 2개가 있음
    - 두 state의 비율을 어느정도로 유지할지에 대한 gate
    - 1에 가까우면 현재 step의 state를 잊고, 0에 가까우면 현재 step만 살림

## LSTM
3개의 gate를 가짐

- Input Gate : cell state에 전달
- Forget Gate : 현재 step 정보 얼마나 기억할지 남김
- Output Gate : 최종적으로 전달할 정보

2개의 Output을 가짐
- h : hidden state
- c : cell state

## Seq2Seq
의미
- 가변적 길이의 Input이 들어왔을 때, 또 다른 가변적 길이의 Output을 예측하늠 모델

특징
- Input을 다루는 Encoder
- Output을 다루는 Decoder
- 2가지가 필요


## Machine Translation
Seq2Seq의 한 분야

구조
- Encoder의 RNN을 적용해 나온 최종 hidden state를
- Decoder의 Input으로 적용해 최초 단어 예측

학습
- Encoder, Decoder에서 특정 단어(state)가 들어갔을 때, 원하는 단어가 나오도록 Softmax
- 원하는 단어와의 Loss를 Cross Entropy형태로 계산

예측
- 이 과정에서는 EOS 토큰이 나올 때 까지 자기 자신을 넣어서 예측

## Attention

기존의 한계
- encoder의 hidden state 하나에 모든 정보가 압축되어 전달됨
- 많은 정보의 손실이 발생

해결 시도
- Encoder의 정보를 Decoding 단계에서 직접 매칭하자.
- 특정 단어와 Encoder의 정보 사이의 연관성을 계산하자 -> Attention

구조
hidden state마다 MLP에 통과시켜 이전 state와 얼마나 어울리는지 계산
- 이를 softmax에 통과시키면 weight matrix $\alpha_i$가 됨
- 이를 hidden vector와 곱해 $c_i$를 구성하고
- 이를 $s_{t-1}$과 concat해 RNN을 통과시켜 $s_{t-1}$을 구성하면
- $s_t$를 이용해서 $y_t$를 예측할 수 있게 됨