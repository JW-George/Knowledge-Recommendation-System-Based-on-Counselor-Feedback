# Knowledge-Recommendation-Processing-System
Knowledge Recommendation System Based on Counselor Feedback by Machine Learning

---

## Abstract
> The editors make great efforts to select appropriate judges for fair and reliable peer review of submitted manuscripts. The editor considers whether the reviewer and authors have a stake in the interest, as well as whether the reviewer has sufficient expertise in reviewing the manuscript. (Omit content) This code proposes a reviewer recommendation algorithm that recommends appropriate reviewers to editors based on the evaluation of these academic activities.

>  편집자는 제출된 원고(manuscript)의 공정하고 신뢰할 수 있는 동료 심사를 위한 적절한 심사자를 선정하는 데 많은 노력을 기울이고 있다. 편집자는 심사자와 저자들 간의 이해 관계 출동 여부는 물론, 심사자가 해당 원고를 심사하는데 있어 충분한 전문성을 갖고 있는지 고려한다. (중간 생략) 본 코드은 이와 같은 학술 활동의 평가에 근거하여 적절한 심사자를 편집자에게 추천하는 심사자 추천 알고리듬을 제안한다.

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
    │  │  │  .DS_Store
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
    │  │  │  .DS_Store
    │  │  │  checkpoint
    │  └─model_4
    │      │  -10000.data-00000-of-00001
    │      │  -10000.index
    │      │  -10000.meta
    │      │  .DS_Store
    │      │  checkpoint
    └─predict_result
            result.csv
```
## Data Pre_Processing

* 입력데이터

bow 작업을 한 csv 파일
'./input_data/index_model.csv'

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

(목적) 입력데이터 path

(타입) string

(예시) path = './input_data/index_model.csv')

* 프로세스

(데이터 나누는 사진 첨부)

Train_Data | Test_data
---|---
80% | 20%

Train(train)_Data | Train(valid)_Data | Test_Data
---|---|---
60% | 20% | 20%

위의 표와 같이 데이터를 자르는 프로세스를 진행합니다.

전체 데이터 중 80프로를 학습에 활용하고,

전체 데이터 중 20프로를 테스트에 활용합니다.

* 출력데이터

X_data,y_data,y_data2

x_train,y_train

x_valid,y_valid

x_test,y_test

x_colum

nb_classes

## Model Training

* 입력데이터

(데이터 전처리 출력데이터)

X_data,y_data,y_data2,x_train,y_train,x_valid,y_valid,x_test,y_test,x_colum,nb_classes

* 입력파라미터

learning_rate = 0.0000005
global_step = 500001
valid_step = 10001
view_step = 5000
saver_step = 10000

* 프로세스

(data_embedding)

(layer_structed)

(loss_function)

(optimizer)

---

(train)




* 출력데이터


## Load Pre_Trained Model

* 입력데이터

* 입력파라미터

* 프로세스

* 출력데이터

## Version History

* v.1.0 : 상담유형 추천 시스템

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
