import streamlit as st
import PIL.Image as Image


def init_value():
    # 스트림릿의 선택 창으로 채점할 문제의 종류를 선택하는 부분입니다.
    year = [str(y) for y in range(2013, 2024)]  # 2013학년도 ~ 2023학년도
    default_ix = year.index("2021")
    year_choice = st.selectbox("채점을 원하시는 시험의 연도를 선택해 주세요.", year, index=default_ix)

    test = ["6월", "9월", "수능"]  # 6월, 9월 모의고사, 수능입니다.
    test_map = {"6월": "6", "9월": "9", "수능": "f"}  # 데이터에 사용된 문자(6, 9, f)로 변환하기 위한 맵입니다.
    test_choice = st.selectbox("채점을 원하시는 시험을 선택해 주세요.", test, index=2)

    type_ = (
        ["확률과 통계", "미적분", "기하"] if int(year_choice) >= 2022 else ["가(A)형", "나(B)형"]
    )  # 가(A)형, 나(B)형, 2022학년도부터 구분 없음(n)
    type_map = {"확률과 통계": "p", "미적분": "c", "기하": "g", "가(A)형": "a", "나(B)형": "b"}
    type_choice = st.selectbox("채점을 원하시는 시험의 종류를 선택해 주세요.", type_)

    return year_choice, test_map[test_choice], type_map[type_choice]
