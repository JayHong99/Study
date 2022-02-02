# AI 504 "Programming for AI"
- KAIST 김재철AI 대학원 소속 Edward Yoonjae choi 교수님의 강의
    - Professor's Page : https://mp2893.com/
- 최신 강의 (2020년 가을)
- 딥러닝 이론 + 코딩으로 구성된 강의
- 다양한 방법론 수업
- 강의자료 : https://mp2893.com/course.html

# Class 19 Image to Text 이론
- More Study Info at : https://jayhong1999.notion.site/AI504-Study-3254606f482e4b7c893a82189275c941

## Image Captioning
Seq2Seq
- Text In, Text Out
- Encoder : RNN
- Decoder : RNN

Image to Seq
- Image in, Text Out
- Encoder : CNN
- Decoder : RNN

## Show and Tell
구조
- 이미지 -> Vision CNN -> RNN
- Encoder 
    - V1 Image Net
- Decoder 
    - Single LSTM
- Total
    - Encoder에서 128x128 이미지
    - 512차원 Output -> LSTM에 Input
    - Beam 20 Search

## Show Atten and Tell
Attention을 적용

구조
- 이미지 Input
- CNN 14x14 feature map
- RNN with Attention
- Word by Word Generation

## Text to Image
- Text In, Image Out

구조
- RNN Encoder
- GAN Decoder
    - Generator : Text
    - Discriminator : Text + Image