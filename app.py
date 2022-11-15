import os

import streamlit as st
from PIL import Image  # 파이썬 기본라이브러리는 바로 사용 가능!


def get_image(image_name):
    image_path = f"{os.path.dirname(os.path.abspath(__file__))}/{image_name}"
    image = Image.open(image_path) # 경로와 확장자 주의!
    st.image(image)

get_image("titanic.png") # https://www.canva.com/

import pandas as pd
import numpy as np

data_url = "https://raw.githubusercontent.com/bigdata-young/bigdata_16th/main/data/titanic_train.csv"
df = pd.read_csv(data_url) # URL로 CSV 불러오기

st.write("타이타닉 탑승자 데이터")
st.write(df.head(10)) # 자동으로 표 그려줌
# st.table(df) # 이걸로 그려도 됨
st.write("데이터 출처")
st.write("https://www.kaggle.com/competitions/titanic/data")

import joblib
model_path = f"{os.path.dirname(os.path.abspath(__file__))}/titanic_model_lr.pkl"
model = joblib.load(model_path)

st.write("---")

# 첫번째 행
r1_col1, r1_col2 = st.columns(2)

# 나이 입력 (숫자 입력)
age = r1_col1.number_input(
    label="나이", # 상단 표시되는 이름
    min_value=0, # 최솟값
    max_value=99, # 최댓값
    step=1, # 입력 단위
    value=20, # 기본값
)

# 성별 입력 (라디오 버튼)
sex_button = r1_col2.radio(
    label="성별", # 상단 표시되는 이름
    options=["남성", "여성"], # 선택 옵션
    # index=0 # 기본 선택 인덱스
    horizontal=True # 가로 표시
)
    
if sex_button == "남성" :
    sex = -1
else :
    sex = 1

# 두번째 행
r2_col1, r2_col2 = st.columns(2)

# 객실 등급 입력 (셀렉트 박스)
pclass_button = r2_col1.selectbox(
    label="객실 등급", # 상단 표시되는 이름,
    options=["1등급", "2등급", "3등급"],
)

if pclass_button == "1등급" :
    pclass = 1
elif pclass_button == "2등급" :
    pclass = 2
else :
    pclass = 3
     
# 객실 번호 입력 (라디오 버튼)
cabin_class_button = r2_col2.radio(
    label="객실 번호",
    options=["있음", "없음"],
    horizontal=True
    )

if cabin_class_button == "있음" :
    cabin_class = 1
else :
    cabin_class = 0
    
# 실행 버튼
play_button = st.button(
    label="예측", # 버튼 내부 표시되는 이름
    )

st.write("---") # 구분선

# 개별 데이터 생존 확률 계산
data = [pclass, sex, age, sex*pclass, cabin_class]
log_odds = sum(data * model.coef_[0]) + model.intercept_[0]
odds = np.exp(log_odds)
p_death = round(((1/(1+odds)) * 100), 2)
p_surv = round((100 - p_death), 2)


# 실행 버튼이 눌리면 모델을 불러와서 예측한다
if play_button :
    input_values = [[
        pclass, sex, age, sex*pclass, cabin_class
        ]]
    pred = model.predict(input_values)
    st.write("예측 결과")
    if pred[0] == 1 :
        st.write("## 생존")
        st.write(f"{p_surv}% 확률로 생존, {p_death}% 확률로 사망")
    else :
        st.write("## 사망")
        st.write(f"{p_surv}% 확률로 생존, {p_death}% 확률로 사망")
    get_image("death.jfif")
    
st.write("---")
    
st.write("#### 로지스틱 회귀 모델")
st.write(pd.DataFrame(model.coef_[0], index=["pclass", "sex", "age", "sex*pclass", "cabin_class"], columns=["계수"]).T)
   
st.markdown(
    """
    ```
    # 개별 데이터 생존 확률 계산
    data = [pclass, sex, age, sex*pclass, cabin_class]
    log_odds = sum(data * model.coef_[0]) + model.intercept_[0]
    odds = np.exp(log_odds)
    p_death = round(((1/(1+odds)) * 100), 2)
    p_surv = round((100 - p_death), 2)
    ```
    """
    )