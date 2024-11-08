import streamlit as st
import importlib
import sys
from PIL import Image
import os

# 먼저 페이지 설정을 실행
st.set_page_config(layout="wide", page_title="Samsung C&T Resort - GenAI Tools")

# POC 파일들의 경로를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# POC 모듈 임포트
import poc_1
import poc_2

def main():
    # 사이드바 설정
    with st.sidebar:
        st.title(":blue[Samsung C&T Resort]")
        st.caption("GenAI Tools")
        
        # 앱 선택 라디오 버튼
        app_choice = st.radio(
            label="생성형 AI 도구 선택",  # label 추가
            options=["Daily VOC Analysis", "YouTube Comments Analysis"],
            key="app_choice"
        )
        
        # 선택된 도구에 대한 설명
        st.markdown("---")
        if app_choice == "Daily VOC Analysis":
            st.markdown("""
            ### 일일 VOC 요약
            - Excel (일일 VOC) 업로드
            - VOC 자동 요약, 분류 및 답변 생성
            - 분석 결과 다운로드
            - 분석 결과 이메일 발송
            """)
        else:
            st.markdown("""
            ### YouTube 댓글 분석
            - YouTube URL 입력
            - 댓글 수집 및 분석
            - 분석 결과 다운로드
            """)
    
    # 메인 컨텐츠 영역
    if app_choice == "Daily VOC Analysis":
        importlib.reload(poc_1)  # 모듈 재로드
        poc_1.main()
    else:
        importlib.reload(poc_2)  # 모듈 재로드
        poc_2.main()

if __name__ == "__main__":
    main()