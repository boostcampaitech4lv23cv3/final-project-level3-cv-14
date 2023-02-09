# data version

## train data

t_v5(available): 필기만 있는 데이터(768장) + 필기, 체크 표시가 있는 데이터(1536장) + 필기, 체크 표시, 주관식이 있는 데이터(768장)<br/>
train_v4(available): base(768장) + 체크 표시만 있는 데이터(768장) + 필기, 체크 표시가 있는 데이터(768*6장), split한 데이터(t_v4[2013-2020], v_v4[2021-2023])<br/>
train_v3: 필기가 없고 체크 표시만 있는 데이터 + 필기와 체크 표시 모두 있는 데이터<br/>
train_v2: 답이 체크되어 있지 않은 데이터에도 필기 추가, 필기의 개수 랜덤하게(1\~40개) 변경, split한 데이터(t_v2[2013-2020], v_v2[2021-2023])<br/>
train_v1-4: 필기와 보기가 겹치지 않는 합성 데이터로 교체<br/>
train_v1-3: 2022, 2023년도 파일 이름에 type 추가 / 풀이가 있는 데이터 추가<br/>
train_v1-2-1: 오류 수정<br/>
train_v1-2: 데이터(2023) 추가<br/>
train_v1-1: 오류 수정<br/>
train_v1: 풀이가 없는 시험지(2013~2022) 데이터 구축<br/>
base(available): 2013 ~ 2023년도 6월 모의고사, 9월 모의고사, 수능 빈 시험지(총 768장)<br/>

<br/>

## 합성 데이터 제작을 위한 데이터
data hw(available): hand writing 데이터(필기 패치 데이터(10,644장) + 체크 표시 데이터(18장))<br/>
- 필기 패치 데이터 출처: <https://www.isical.ac.in/~crohme/>
- 체크 표시 데이터: 직접 제작<br/>

mnist_ocr(available): ocr task를 위한 숫자 데이터(0~999 사이 숫자 데이터 10,000장)
- mnist 활용
- 출처: <http://yann.lecun.com/exdb/mnist/>

<br/>

## annotation data

t_v5, v_v5(available): 주관식을 위한 class 추가<br/>
train_v4-1(available): q(question) class 추가(train_v4 data 사용)<br/>
train_v4(available): train_v4 data에 해당하는 annotation data

<br/>

## test data

test_v1: 풀이가 없는 시험지(2012) 데이터<br/>
test_v2: 풀이가 있는 시험지 데이터(4세트)<br/>

<br/>

## file naming

**file name: {year}\_{test}\_{type}_{풀이여부}-{쪽수}.jpg**
* test: 6월 모의고사, 9월 모의고사, 수능(6, 9, f)
* type: 가형, 나형 또는 a형, b형 / 2022학년도부터는 통합(a, b, n)
* 풀이여부: 풀이가 없는 시험지(0), 풀이가 있는 시험지(1, 2, ...)

<br/><br/>

# data load

`dvc pull {train_v1.dvc} {test_v1.dvc} {train_v1.json.dvc} {test_v1.json.dvc}`
