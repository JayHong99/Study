# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 15 Word Embedding
- More Study Info at : https://jayhong1999.notion.site/AI504-Study-3254606f482e4b7c893a82189275c941

## One-Hot Encoding
하나만 1이고, 나머지는 0으로 만드는 방법
- Example)
    - Cat, Kitty, Dog, Puppy
    - Cat : [1,0,0,0]
    - Kitty : [0,1,0,0] ...
- 특징
    - 해석하기에 용이
    - 단어사이의 내적은 항상 0, Eucldean 거리는 항상 root 2
        - 즉, 단어와 단어 사이의 유사성을 계산하지 못함
    - 차원의 낭비가 너무 심함

## Vectorization
### Term-Document Matrix
모든 단어를 개별 Token으로 바라보고, 해당 Token이 몇 번 나왔는지 세는 방법

특징
- 두 문장 사이의 유사도를 구할 수 있음
    - 비슷한 문장은 비슷한 단어의 반복 횟수를 가짐
    - 그 문장들을 내적후 합하면 유사도에 대한 값이 됨
    - 그 외에도 Cosine Similarity 등을 적용 가능

### Matrix Factorization
SVD 를 적용한 방법

### Probabilistic Topic Modeling
LDA를 적용한 방법

### Word Embedding

### Distributed Vector Representation
- Vector를 다차원 공간속에서 Non-zero하게 만드는 방법
- 이러면 두 단어 사이의 의미론적 해석을 내적, cosine 유사도 등을 이용해 가능

## Word 2 Vector
Idea
- 중심단어 c는 그 주변단어 o로 인해 구성된다.
- 단어는 비슷한 문맥 속에서 비슷한 의미를 형성한다.

특징
- One-Hot이 해당 단어를 정확히 나타낸다면, W2V는 의미 자체는 희미해짐 (추정값)
- 의미를 선형적 해석하기에 좋음
    - Vector는 방향과 크기를 가짐
    - 방향성을 이용해서 선형적 해석이 용이
    - Man -> Woman ~ King -> Queen

### Skip-gram
Idea
- 중심단어 c를 이용해서 문맥단어 o를 구성할 수 있다.
- 기존 W2V와 다른 접근법

학습
- 모든 단어에 대해서, 중심단어 c가 나타났을 때, 문맥단어 o가 나타날 확률을 계산 
    - 모든 단어에 대해서 계산해야 하기 때문에 학습속도가 느림
    - 이를 모든 문장 속 모든 단어에 대해서 적용하기에 더욱 느림
- CBOW (기존 W2V)보다 학습 속도가 훨씬 느림
- 의미론적 해석에서 좋은 성능을 가짐

학습 시간 보완
- Hierarchical Softmax -> 2진 분류 방식으로 Update (거의 사용 X)
- Negative Sampling -> K개의 단어만을 sampling해 계산 + 이진 분류 문제로 접근
    - 같은 문맥속에서 나타나는 단어들 -> True Case of Binary Classification
    - 다른 문맥속에서 나타나는 단어들 -> False case of Binary Classification
    - 이를 이용해서 negative log likelihood (loss를 minimize하기 위해)계산

- 단어가 나타내는 vector만 업데이트하는 방법

### W2V 적용 분야
- 단어간 유사도 계산
- 기계 번역
- POS, NER
- 감성 분석
- Clustering
- Semantic Lexicon Building

## GloVe
- Global Vectors for Word Representation
- 동시 발생 matrix를 이용해서 word embedding
    - 학습 속도 향상
    - 적은 단어로도 잘 학습

## Doc2Vec
- 이제는 W2V와 GloVe를 잘 사용 X
- Document Vector도 하나의 단어로 취급하자.

특징
- 같은 문맥이나 문장에 나타나는 단어는 유사도가 높아짐.
- Document가 같은 공간에 Embedding 될 수 있음
    - 다의어 차원에서 장점을 가짐

## Sub-word Embedding
### 문자체계
언어마다 문자 체계가 너무 다름
- 중국어, 일본어는 띄어쓰기가 없는 등

단어는 변형이 가능함
- Good -> gooooooood 등

### Character - Level Models
이를 보완하기 위한 모델

Character 단위로 tokenization해서 학습하자.

### Byte Pair Encoding (BPE)
compression algorithm
- 많이 나오는 단어들을 하나로 압축하는 방법

character 단위로 vocabulary를 시작
- 가장 빈번하게 나오는 n-gram pair를 하나의 n-gram으로 만듦
- {아침}과 {밥}이 함께 가장 많이 나온다면 {아침밥}이라는 token을 추가

## Word Piece / Sentence Piece Model
빈도가 아닌 혼란도를 줄이는 n-gram을 추가하는 방법
- 공백을 _로 대체하는 등 CJK (Chinese Japanese Korean) 공용 방법
- BERT는 Word Piece를 사용

## Contextualized Word Embedding
단어의 의미는 문맥에 따라 변화
- Q) "배"의 Embedding 값을 알려줘
- A) 어떤 맥락에서의 "배"인가?
- Q) "배가 많이 나와서 운동좀 해야겠어"
- A) 그 맥락에서는 ~~~이야.

