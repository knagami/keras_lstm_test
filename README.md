# keras_cnn
## 目的
LSTM(RNN)のサンプル

## 参考ページ
https://qiita.com/everylittle/items/ba821e93d275a421ca2b
http://cedro3.com/ai/keras-seq2seq/
http://cedro3.com/ai/keras-lstm-text-word/

## 使い方

1. 以下のコマンドで、固定長の入力データで学習 
>python3 createModel.py

2. 以下のコマンドで、可変長の入力データで学習 
>python3 createModel2.py

3. 以下のコマンドで、 seq2seqで英日翻訳
>python3 createModel3.py

4. 以下のコマンドで、 テキストをUTF8に変更
>python3 createModel4_0.py

以下のコマンドで、 文章生成を単語単位で実施
>python3 createModel4.py