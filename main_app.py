import os
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

# 페이지 설정
st.set_page_config(page_title="ChatPDF")

def display_messages():
    """
    화면에 사용자와 어시스턴트의 대화 메시지를 표시하는 함수.
    """
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """
    사용자의 입력을 처리하고 어시스턴트의 응답을 화면에 추가하는 함수.
    """
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        # 어시스턴트가 응답을 생성하는 동안 스피너 표시
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
        # 사용자와 어시스턴트의 메시지를 세션 상태에 추가
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def page():
    """
    Streamlit 페이지의 주요 로직을 처리하는 함수.
    """
    if len(st.session_state) == 0:
        # 세션 상태 초기화
        st.session_state["messages"] = []
        # ChatPDF 인스턴스 생성 시 추출된 텍스트 파일 경로 전달
        text_file_path = "data/extracted_text.txt"
        st.session_state["assistant"] = ChatPDF(text_file_path)

    # 페이지 헤더 설정
    st.header("ChatPDF")
    # 대화 메시지 표시
    display_messages()
    # 사용자 입력 텍스트 박스
    st.text_input("Message", key="user_input", on_change=process_input)

# 메인 함수 실행
if __name__ == "__main__":
    page()

#streamlit run main_app.py
