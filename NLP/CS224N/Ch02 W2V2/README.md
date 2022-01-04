# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# W2V2 (2021. 12.19)

## W2V Review
- 중심 단어를 사용해 문맥 단어를 추측한다.

## Problem of W2V
- theta Update 비용이 너무 큼 (모든 단어들에 대해서 Update하기 힘듬)
    -> SGD -> 등장하는 단어들만 update하자 -> Word Embedding의 관점에서 가능
- 문장 수의 부족 혹은 낭비
    -> 문맥 단어를 이용해서 중심단어를 예측하자 (CBOW)
- Update할 때마다 Softmax하는 것이 너무 비싸다 
    -> Negative Sampling -> Center : Context가 같이 등장할 가능성에 대해서 binary classificaiton의 관점으로 접근한다.

## Co-occurrence
Full Document
- 모든 단어들에 대해서 단어 등장 횟수 matrix를 계산
    -> high dim & sparse prob -> Low- Dimensional Vectors

Low - Dimensional Vectors(SVD)
- 중요한 단어는 25 ~ 1000 차원의 dense vector로 표현 가능할 것이다.
    -> SVD 로 표현
    -> Scailing 하면 성능이 더 좋을 것이다.

SVD - S
- SVD 에 Scailing 적용
    -> Scailing을 더 많이 적용하자.

SVD - L (COALS)
- SVD에 다음 3가지 적용
    - Log 변환
    - Min(x,t) = 100
    - 구조적 역할 (a, the, ... ) 제거

## GloVe
Log-bilinear model
- wx(wa-wb)
    - 단어 x에 대해서 단어 a가 등장할 비율을 단어 b가 등장할 비율로 나눈 값에 로그를 취한 값이다.
    -> 이 값이 크면 동시에 등장할 가능성이 높은 것이고, 낮으면 동시에 등장할 가능성이 낮음을 의미
    -> 즉, 단어 x에 대해서 단어 a와 b가 동시에 등장할 가능성을 측정
    -> 많은 단어 x에 대해서 측정하면, a와 b가 동시에 등장할지 측정 가능 

## 모델 성능 측정
Intrinsic
- 벡터 사이의 합차를 이용해 본질적으로 계산
- 유사어, 반의어를 얼마나 맞추는지 측정
- WS353같은 인간이 세운 기준에 얼마나 부합하는지 유사도 측정도 가능

Extrinsic
- Word Vector를 잘 만들었다면, NER의 관점에서 결과가 좋을 것이다.

성능을 높이는 법
- More data
- Use Good data
- Set Good Word Dimmension

## Word Sense
단어 하나도 다양한 뜻을 지님 -> 이를 Vector로 어떻게 표현?
- 하나의 Vector를 여러개로 쪼개어 단어_1 , 단어_2 ... 이런식으로 표현
- 단어는 통합해서 Embedding Vector로 표현되지만, 쪼개면 같은 단어 속 다른 의미들을 grouping 할 수도 있음
- example) tie : 신발끈, 리본 => 외모적인 부분 vs 동점 => 운동적인 부분 등