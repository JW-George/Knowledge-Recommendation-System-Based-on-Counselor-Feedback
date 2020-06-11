# Knowledge-Recommendation-Processing-System
Knowledge Recommendation System Based on Counselor Feedback by Machine Learning

---
<p align="center">
  <br>
  <img src="./image/spelix.png"><br>
  <a href="#">SPELIX Inc. R&D Center</a> |
  <a href="#">AI Team</a>
</p>


## Abstract
> The ordering company is putting a lot of effort into establishing an appropriate knowledge recommendation system for an accurate and prompt knowledge recommendation system of counseling services. The ordering company considers whether it has rule-based knowledge recommendation technology as well as machine learning-based knowledge recommendation technology in selecting an operator. This system is a knowledge recommendation system that recommends the appropriate knowledge type to the counselor based on the feedback from the counselor.

> 발주사는 상담서비스의 정확하고 신속할 수 있는 지식 추천 시스템을 위한 적절한 지식추천 시스템을 구축하는데 많은 노력을 기울이고 있다. 발주사는 시행사를 선정하는데 있어 룰기반(Rule Based) 지식추천 기술은 물론, 기계학습(Machine Learning)기반 지식추천 기술을 갖고 있는지를 고려한다. 본 시스템은 이와 같이 상담사 피드백에 근거하여 적절한 지식유형을 상담사에게 추천하는 기계학습(Machine Learning)기반 지식추천 시스템입니다.

## System Structure
```bash
│  .gitignore
│  LICENSE
│  README.md
└─system
    │  train_and_load.ipynb
    ├─input_data
    │      .DS_Store
    │      index_model.csv
    ├─model
    │  ├─model_0
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_1
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_2
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_3
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  └─model_4
    │      │  -10000.data-00000-of-00001
    │      │  -10000.index
    │      │  -10000.meta
    │      │  checkpoint
    └─predict_result
            result.csv
```
## Data Pre_Processing

* 입력데이터

```bash
    ├─input_data
    │      .DS_Store
    │      index_model.csv
```

(설명) BOW(Back of Word)을 통해 keyword의 등장 유/무를 통해 STT를 정형화한 csv 파일\
(용도) x_data | y_data | train | valid | test 등 데이터 분할\
(타입) csv\
(예시) './input_data/index_model.csv'

* 입력 데이터 구조

keyword_1 | keyword_2 | keyword_3 | ... | keyword_1444 | Label |  
---|---|---|---|---|---
1 | 0 | 0 | ... | 0 | 0
0 | 1 | 0 | ... | 0 | 1
0 | 0 | 1 | ... | 0 | 2
... | ... | ... | ... | ... | ...
0 | 0 | 0 | ... | 1 | 25

(Shape : (row)1409 x (col)1445, Type : CSV)

* 입력 파라미터

(명명) path\
(타입) str\
(목적) 입력데이터 고정 path 지정\
(예시) path = './input_data/index_model.csv'

* 프로세스

![data](/image/data.png)

Train_Data | Test_data
---|---
80% | 20%

Train(train)_Data | Train(valid)_Data | Test_Data
---|---|---
60% | 20% | 20%

위의 표와 같이 데이터를 자르는 프로세스를 진행합니다.

전체 데이터 중 80프로를 학습(train)에 활용하고,

전체 데이터 중 20프로를 테스트(test)에 활용합니다.

* 출력데이터

X_data | y_data | y_data2
---|---|---
keyword1,2,...,1444 | lable(unknown) | lable(original)

> lable(original) : 원래 lable 값(1~25)\
lable(unknown) : unknown 처리를 위해 원래 lable 값 5이상 값 5로 변환

x_train | x_valid | x_test
---|---|---
keyword1,2,...,1444(60%) | keyword1,2,...,1444(20%) | keyword1,2,...,1444(20%)

y_train | y_valid | y_test
--- | --- | ---
lable(original, 60%) | lable(original, 20%) | lable(original, 20%)

x_colum | nb_classes
--- | ---
keyword colum | 중복 제거 lable 수

## Model Training

* 입력데이터

```python
input_of_train = data_preprocessing()
```

(명명) X_data,y_data,y_data2,x_train,y_train,x_valid,y_valid,x_test,y_test,x_colum,nb_classes\
(타입) Pandas.DataFrame\
(목적) '데이터 전처리' 리턴 값를 학습에 입력함\
(예시) X_data,y_data,y_data2,x_train,y_train,x_valid,y_valid,x_test,y_test,x_colum,nb_classes

* 입력파라미터

```python
learning_rate = 0.0000005
global_step = 500001
valid_step = 10001
view_step = 5000
saver_step = 10000
```

(명명) learning_rate,global_step,valid_step,view_step,saver_step\
(타입) int | float\
(목적)\
learning_rate : (학습률/학습보폭) local minimun 도달을 위한 기울기에 곱해지는 스칼라 값\
global_step : x_train|y_train 학습 epoch\
valid_step : x_valid|y_valid 학습 epoch\
view_step : 입력값 만큼의 학습진행 상황을 시각화\
saver_step : 입력값 만큼의 학습진행 상황을 모델화(checkpoint지정)\
(예시)\
learning_rate = 0.0000005\
global_step = 500001\
valid_step = 10001\
view_step = 5000\
saver_step = 10000

* 프로세스

1. data_embedding

```python
X, y, target,Y_one_hot = data_embedding(nb_classes,x_colum)
```
tf.placeholder, reshape을 통한 data embedding 작업

2. layer_structed

```python
W5, b5, layer_5, y_pred, keep_prob = layer_structed(X, y, target, nb_classes, x_colum)
```

![layer](/image/layer.png)

Activation function : sigmoid function\
x_colum 갯수 만큼의 input node\
nb_classes 갯수 만큼의 output node

3. loss_function

```python
loss = loss_function(target,y_pred)
```
Loss function : tf.reduce_mean

4. optimizer

```python
d_b, d_W = optimizer(y_pred, target, layer_5, X)
```
optimizer : Back propagation

5. train & save

```python
train(x_train,y_train,x_test,y_test,x_valid,y_valid,nb_classes,x_colum)
```

train : sess.run(tf.global_variables_initializer())
save : saver.save(sess, path, global_step)

* 출력데이터

```bash
    ├─model
    │  ├─model_0
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_1
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_2
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_3
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  └─model_4
    │      │  -10000.data-00000-of-00001
    │      │  -10000.index
    │      │  -10000.meta
    │      │  checkpoint
```

(명명) -10000.data-00000-of-00001,-10000.index,-10000.meta,checkpoint\
(타입) tensorflow.model\
(목적) '데이터 전처리' 리턴 값를 학습에 입력함

## Load Pre_Trained Model

* 입력데이터

데이터전처리 csv
```python
input_of_train = data_preprocessing()
```

(명명) X_data,y_data,y_data2,x_train,y_train,x_valid,y_valid,x_test,y_test,x_colum,nb_classes\
(타입) Pandas.DataFrame\
(목적) '데이터 전처리' 리턴 값를 학습에 입력함\
(예시) X_data,y_data,y_data2,x_train,y_train,x_valid,y_valid,x_test,y_test,x_colum,nb_classes

사전학습 모델
```bash
    ├─model
    │  ├─model_0
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_1
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_2
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  ├─model_3
    │  │  │  -10000.data-00000-of-00001
    │  │  │  -10000.index
    │  │  │  -10000.meta
    │  │  │  checkpoint
    │  └─model_4
    │      │  -10000.data-00000-of-00001
    │      │  -10000.index
    │      │  -10000.meta
    │      │  checkpoint
```

(명명) -10000.data-00000-of-00001,-10000.index,-10000.meta,checkpoint\
(타입) tensorflow.model\
(목적) '데이터 전처리' 리턴 값를 학습에 입력함

* 입력파라미터

```python
new_df=load(nb_classes,x_colum,X_data,y_data,y_data2,
     path='./predict_result/result.csv')
```

(명명) path\
(타입) str\
(목적) 출력결과 export 고정 path 지정\
(예시) path='./predict_result/result.csv'

* 프로세스

1. data_embedding

```python
X, y, target,Y_one_hot = data_embedding(nb_classes,x_colum)
```

2. layer_structed

```python
W5, b5, layer_5, y_pred, keep_prob = layer_structed(X, y, target, nb_classes, x_colum)
```

3. restore

```python
    model_0 = pred_by_restore('./model/model_0',W5, b5, layer_5, y_pred, keep_prob, Y_one_hot,X_data,X,y)
    model_1 = pred_by_restore('./model/model_1',W5, b5, layer_5, y_pred, keep_prob, Y_one_hot,X_data,X,y)
    model_2 = pred_by_restore('./model/model_2',W5, b5, layer_5, y_pred, keep_prob, Y_one_hot,X_data,X,y)
    model_3 = pred_by_restore('./model/model_3',W5, b5, layer_5, y_pred, keep_prob, Y_one_hot,X_data,X,y)
    model_4 = pred_by_restore('./model/model_4',W5, b5, layer_5, y_pred, keep_prob, Y_one_hot,X_data,X,y)
```

4. predict ranking export

```python
model_list = hyun2(model_0,model_1,model_2,model_3,model_4)
```

5. save_csv2

```python
new_df = save_csv2(path,y_data2,model_list)
```

6. print_predict

```python
print_predict(new_df)
```

* 출력데이터

```bash
    └─predict_result
            result.csv
```

rank0 | rank1 | rank2 | rank3 | rank4 | rank5 | real_Y | bool_result
---|---|---|---|---|---|---|---
0 | 3 | 23 | 1 | 4 | 2 | 0 | True
... | ... | ... | ... | ... | ... | ... | ...
7 | 8 | 5 | 9 | 3 | 0 | 0 | False

(명명) result.csv\
(타입) csv\
(설명) 시스템 예측 & 원래 정답 Label 비교 결과 정리한 csv\

## Version History

* v.1.0 : Global Train Epoch 500,000
* v.1.1 : ADD API with empty list
* v.1.2(to do) : Extract Keyword BOW
* v.1.3(to do) : 이진 분류 as konlpy
* v.1.4(to do) : re-training as konlpy

## Contacts

현진우, hyunjw0123@gmail.com

> SPELIX Inc. R&D Center | AI Researcher

박태수, tessinu@spelix.com

> SPELIX Inc. R&D Center | AI Researcher


## License

* 본 코드는 원저자의 정책과 동일하게 Apache-2.0 라이선스 하에 공개되어 있습니다.
* 본 코드는 비상업적 목적으로 한정합니다. 상업적인 목적을 위해서는 위의 Contacts로 연락해 주세요.
* 본 코드에 추가/수정한 코드의 경우에도 비상업적 목적으로 한정합니다. 상업적인 목적을 위해서는 위의 Contacts로 연락해 주세요.
* 심사자 추천 기술 적용 및 협업 등의 문의를 환영합니다.
