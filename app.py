import os

import streamlit as st
from PIL import Image  # 파이썬 기본라이브러리는 바로 사용 가능!


def get_image(image_name):
    image_path = f"{os.path.dirname(os.path.abspath(__file__))}/{image_name}"
    image = Image.open(image_path) # 경로와 확장자 주의!
    st.image(image)

get_image("titanic.png") # https://www.canva.com/

import pandas as pd  # 판다스 불러오기

data_url = "https://raw.githubusercontent.com/bigdata-young/bigdata_16th/main/data/titanic_train.csv"
df = pd.read_csv(data_url) # URL로 CSV 불러오기

st.write(df) # 자동으로 표 그려줌
# st.table(df) # 이걸로 그려도 됨

import joblib
model_path = f"{os.path.dirname(os.path.abspath(__file__))}/titanic_model_lr.pkl"
model = joblib.load(model_path)
st.write("## 로지스틱 회귀 모델")
st.write(pd.DataFrame(model.coef_[0], index=["pclass", "age", "sex", "sex*pclass", "cabin_class"], columns=["계수"]))

st.write("---")

pclass = st.selectbox(
    label="객실등급", # 상단 표시되는 이름,
    options=[1, 2, 3]
)

with st.echo(code_location="below"):
    # 나이 입력 (숫자)
    age = st.number_input(
        label="나이", # 상단 표시되는 이름
        min_value=0, # 최솟값
        max_value=99, # 최댓값
        step=1, # 입력 단위
        # value=30 # 기본값
    )
    
with st.echo(code_location="below"):
    # 성별 입력 (라디오 버튼)
    sex_button = st.radio(
        label="성별", # 상단 표시되는 이름
        options=["남성", "여성"], # 선택 옵션
        # index=0 # 기본 선택 인덱스
        # horizontal=True # 가로 표시 여부
    )
    
if sex_button == "남성" :
    sex = 0
else :
    sex = 1

with st.echo(code_location="below"):
    cabin_class_button = st.radio(
        label="객실번호",
        options=["있음", "없음"]
    )

if cabin_class_button == "있음" :
    cabin_class = 1
else :
    cabin_class = 0
    
with st.echo(code_location="below"):
    # 실행 버튼
    play_button = st.button(
        label="예측", # 버튼 내부 표시되는 이름
    )

st.write("---") # 구분선

with st.echo(code_location="below"):
    # 실행 버튼이 눌리면 모델을 불러와서 예측한다
    if play_button:
        st.snow() # 눈송이 애니메이션 표시
        input_values = [[
            pclass, age, sex, sex*pclass, cabin_class
        ]]
        pred = model.predict(input_values)
        st.write("## 분류")
        st.write("생존" if pred[0] == 1 else "사망")


