# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# Dependency Parsing (2021. 12.21)

## Linguistic Structure
영어의 문장 구조에 대한 다양한 이야기
중의적 표현에 대한 문제 제기
문장의 구조를 분석하는 방법

## Dependency Treebanks
Dependency Tree
사람이 직접 만든 문장 구조 나무
4가지 조건에 맞추어 문장 구조를 형성
인간의 노력이 많이 들어감

Dependency Treebanks
해당 나무를 저장한 창고
나중에 다시 사용할 수 있어 애용됨

## Dependency Parsing Methods
4가지 중 "Transition-based Parsing" 이용

stack, buffer, arcs A의 구조와 Shift, Left-arc, Right-arc의 패턴을 조합해 문장의 의존 구조를 분석하는 방법

Malt Parser
위의 방법을 통해 만들어진 문장 구조에 대해 Support Vector Machine을 적용해 문장 구조 예측하는 방법 진행
Beam Search를 통해 root 단어 다음에 나올 단어의 문장 구조를 여러개의 가능성을 두고 분석

Evaluation : Accuracy
정답지 : 사람이 만든 정답지
1) UAS : Unlabeled Attachment Score : parsing 순서를 얼마나 맞췄느냐
2) LAS : Labeld Attachment Score : parsing 순서 x 의존 형태 (noun subject 등)
