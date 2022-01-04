# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# Transformers + Pretraining (2021. 12.26)

기존 transformer
- W2V Embedding => Attention Layer
- W2V Embedding에 문제점?
    - "배" 와 같은 다의미적 단어 구분이 어려움
    - UNK등의 단어가 많음
- W2V 문제 해결책
    - UNK => Subword로 Byte 단위로 쪼개어 Vocab을 구성해서 학습
    - 다의미 => Transformer에서 Input 부터 아예 학습

## Transformers Pretraining
### Decoder Pretraining
Example) GPT
특징
- Decoder는 미래의 단어를 사용해서 학습하면 안됨
- 미래 단어를 Masking 처리하며 차례로 예측하며 학습

### Encoder Pretraining
Example) BERT
Encoder는 Bi-directional Context를 파악하는게 좋음
=> 단어를 Masking 처리하고, 전후 맥락을 모두 살펴 masking 된 단어를 예측 (skip-gram과 유사)
- BERT의 Masking 처리 방법
    - 문장마다 15%의 단어에 Masking을 처리
    - 15%의 80%는 그대로 Masking 유지
    - 15%의 10%는 Random Token으로 변경
    - 15%의 10%는 정답 Token 사용
- BERT의 추가 Embedding
    - Token Embedding : Token에 대한 정보
    - Segment Embedding : 해당 Token이 몇번째 문장인지에 대한 정보
    - Position Embedding : 단어의 번호 정보
- Generation에서는 별로..

### Encoder-Decoder Pretraining
Example) T5
아예 통채로 학습하자
-> 성능이 좋아짐

