# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# W2V (2021. 12.19)

## Language의 미학을 Computer가 학습하게 할 수 없을까?
- Langugage의 미학
    - 언어적 표현
    - 비언어적 표현
    - 문맥
    -> etc... 등에 의해 항상 문장이 나타내는 의미가 다름

## WordNet
- 단어와 단어사이의 관계를 인간이 기록
    - Human effort
    - Subjective
    - Word Similarity
    - Missing Nuance
    => Discrete Word

## Discrete word
- One-Hot Value로 표현
    - 모든 단어에 대해 One-Hot하게 표현 -> High Dimmensional
    - Word Similarity 표현 여전히 X
        - 단어 검색에서 유사단어 찾기가 안됨
    -> Word2Vector

## Word2Vector
"단어는 비슷한 문맥에서 형성된다"
- 단어가 나오는 문장을 많이 수집해서, 단어를 학습하자
- 중심단어 c에 대해서, 문맥단어 o가 나올 확률을 학습
- 수학적 해석
    - P(o|c) : 중심단어 c가 나왔을 때 문맥단어 o가 나올 확률
    - Objective Fucnction : exponential(중심단어 vecotr * 문맥단어 vector) -> softmax
        - 모든 단어들에 대해서 해당 단어가 나타날 확률을 계산
    - Optimize Function : 
        - 중략 (Notion 참고) => Observed - Prediction : 실제값  - 예측값
        - 실제인 값과 예측값의 차이를 Loss로 Optimization을 진행
- 특징
    - Vector이다보니
        - 비슷한 단어는 비슷한 값을 지님
        - 비슷한 값은 비슷한 방향성을 지님
        - example ) king - man + queen = woman
        - 이처럼 방향성과 값을 조합해 새로운 단어를 찾아나갈 수 있음
