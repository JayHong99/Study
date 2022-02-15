# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 23 BERT & GPT

## Word Embedding
### Review
- 단어를 Vector로 표현하는 방법
- 비슷한 단어는 짧은 거리
- 다른 단어는 다른 vector 표현
- CBOW : 주변단어로 중심단어 예측
- Skip-gram : 중심단어로 주변단어 예측

### Contextualized Word Embedding
- Context에 대한 Word Embedding
    - 문맥마다 말하는 단어의 의미가 다름 -> 다른 Embedding Vector를 가져야 함
    - 이전 Word Embedding(W2V , Glove등)은 고정된 단어 의미를 지님

### ELMo
- Bi-LSTM을 이용한 Embedding
- Downstream Task에 맞게 Embedding을 학습

## GPT & BERT
### GPT-1
- Transformer 형태에서 Decoder구조를 사용
- 12 layer 768 hidden size
- 성능이 좋아짐

### BERT
- Bidirectional Encoder 사용
- ELMo와 GPT에서 영감
- Token Embedding + Positional Embedding + trainable Segment Embedding 사용
    - 단어 Embedding : 단어에 대한 Vector추출
    - Positional Embedding : 위치 정보 (몇번째 단어?)
    - Segment Embedding : 문장 정보 (문장 구분)
- Pre-training
    - Masked language Model : 15%의 단어는 Masking해서 주변단어를 이용해 학습
    - Next Sentence Prediction
        - CLS를 이용해 문장 관계 학습
- 다양한 분야에서 많이 좋아짐

### GPT-2
- 2019년, GPT-1의 10배 크기
- Zero-shot Task에 적용
    - 학습없이 바로 예측하는 Task

### GPT-3
- 2020
- 많은 시간과 노력, 연구 투자
- GPT-2의 100배, 1회 학습에 40~50억원