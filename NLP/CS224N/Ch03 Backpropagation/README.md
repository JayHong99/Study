# CS224N - Stanford Univ. NLP (2021. Winter) 
## Reference : 
- http://web.stanford.edu/class/cs224n/
- https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
- https://jayhong1999.notion.site/CS224N-Stanford-Univ-NLP-2021-Winter-1dd9041a4f514b5e959235fee4b24050

1. Watch Youtube Course (V)
2. Organize Contents and Upload Annotated PDF file on my Notion (V)
3. Summarize on my Github (V)


# Backpropagtaion (2021. 12.21)

## Introduction
- NLP에서 자주 사용되는 예제 NER -> 처리 과정 -> SGD로 Loss 계산 -> Back Propagation 사용한다! 이게 뭘까?

## Matrix Calculus
- Back Prop. 에 사용되는 matrix calcus와 관련된 내용
- Matrix의 미분에 활용되는 Jacobian Matrix와 Calculus
    - Jacobian Matrix를 미분하면, diag(f'x) 가되어 계산에 유용
    - 형태가 다를 경우 shape convention을 적용
        - 이 때, 처음부터 shape convention을 적용하고 한다 -1
        - 최대한 그냥 계산하고, 최후에 shape convention을 적용한다 -2
        - 이 두가지가 혼용되고 있다.

## Back Propagation
- Feed Forward Network는 단순하게 계산 결과를 뒤로 전달
- Back Prop.은 계산 과정을 거꾸로 미분을 거쳐 전달
    - 이 때 Jacobian Matrix를 많이 사용
    - 최근의 많은 DL Frame Works는 이를 제공