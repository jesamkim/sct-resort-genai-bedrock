import streamlit as st
import boto3
import json
from PIL import Image
import pandas as pd
import datetime
from botocore.client import Config

# Amazon Bedrock 클라이언트 설정
bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-west-2")  # Claude 3.x 리전
bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 3})
bedrock_agent_client = boto3.client("bedrock-agent-runtime", 
                                  config=bedrock_config, 
                                  region_name="us-east-1")  # KB 리전

# Knowledge Base ID 설정
kb_id = "ACSN5MRWK2"  ## KB ID 변경 필요

def call_bedrock_model(prompt, model_id):
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        inferenceConfig={
            "maxTokens": 4096,
            "temperature": 0.0,
            "topP": 0
        }
    )
    
    if 'output' in response and 'message' in response['output']:
        return response['output']['message']['content'][0]['text']
    else:
        return "응답 생성에 실패했습니다."

def retrieve(query, kbId, numberOfResults=10):
    return bedrock_agent_client.retrieve(
        retrievalQuery={
            'text': query
        },
        knowledgeBaseId=kbId,
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': numberOfResults,
                'overrideSearchType': "HYBRID"
            }
        }
    )

def get_contexts(retrievalResults):
    contexts = []
    for retrievedResult in retrievalResults:
        contexts.append(retrievedResult['content']['text'])
    return contexts


def process_voc(df):
    results = []
    model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"  ## Claude 3.5 Sonnet (CRIS)
    
    # 진행 상황을 표시할 progress bar 생성
    progress_bar = st.progress(0)
    status_text = st.empty()  # 현재 처리 현황을 표시할 공간
    total_rows = len(df)
    
    for idx, row in enumerate(df.iterrows()):
        voc_text = row[1]['질문내용']
        current_time = datetime.datetime.now()
        
        # 현재 진행 상황 업데이트
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"VOC 분석 중... ({idx + 1}/{total_rows} 건 처리 완료)")
        
        # RAG 검색 수행
        response = retrieve(voc_text, kb_id, 10)
        retrievalResults = response['retrievalResults']
        contexts = get_contexts(retrievalResults)
        
        # 답변 생성을 위한 프롬프트
        answer_prompt = f"""
        Human: 새로운 VOC(Voice of Customer)와 유사한 항목을 찾고, 관련 정보와 담당 부서를 제공하는 것이 당신의 임무입니다.
        다음의 맥락 정보를 사용하여 <question> 태그 안에 있는 질문에 답하세요. 맥락에는 이전 VOC 항목들의 "질문내용", "답변내용", "담당 부서" 등의 정보가 포함되어 있습니다.

        <context>
        {contexts}
        </context>

        <question>
        {voc_text}
        </question>

        다음 형식으로 응답해 주세요:
        이러한 유형의 문의나 이슈를 처리하는 답변내용을 참조하세요.
        답변 내용 및 스타일을 참조하여 고객 질의에 대한 답변을 생성하세요. 생성된 답변만 출력하세요.
        
        예시 응답:
        "안전환경그룹고객안전 *** 프로안녕하세요안전그룹 *** 프로 입니다사고자 모 통화병원 치료 흉터 남을 것 같다고 하며 여자아이라 걱정하심보호자 책임도 있다고 공감하시어 당사에서 도의적인 치료비 지원 예정 입니다문제 없도록 종결 하겠습니다감사 합니다"

        A:"""
        
        # 담당 부서 지정을 위한 프롬프트
        dept_prompt = f"""
        Human: 당신은 리조트 회사의 고객 서비스를 전문으로 하는 AI 어시스턴트입니다. 새로운 VOC(Voice of Customer)와 유사한 항목을 찾고, 관련 정보와 담당 부서를 제공하는 것이 당신의 임무입니다.
        다음의 맥락 정보를 사용하여 <question> 태그 안에 있는 질문에 답하세요. 맥락에는 이전 VOC 항목들의 "질문내용", "답변내용", "담당 부서" 등의 정보가 포함되어 있습니다.

        <context>
        {contexts}
        </context>

        <question>
        {voc_text}
        </question>

        다음 형식으로 응답해 주세요:
        이러한 유형의 문의나 이슈를 처리하는 담당 부서 이름만 답변에 출력하세요.
        추가 설명이나 주석 없이 분류 단어만 작성해주세요.
        
        예시 응답:
        리조트 주토피아운영그룹 동물보전1

        A:"""
        
        result = {
            '날짜 및 시간': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            '질문내용': voc_text,
            '요약': call_bedrock_model(f"""
                                     다음 텍스트의 핵심 내용을 정확하고 간결하게 요약해주세요: 
                                    <text>
                                    {voc_text}
                                    </text>
                                    
                                    요약 지침:
                                    1. 반드시 30자 이내로 작성하세요.
                                    2. 가장 중요한 정보만 포함하세요.
                                    3. 완전한 문장으로 작성하지 말고, 핵심 키워드나 구문만 사용하세요.
                                    4. 불필요한 조사, 접속사는 생략하세요.
                                    5. 원문의 주요 의도나 감정을 반영하세요.
                                    
                                    응답 형식은 다음 예시 <example></example>를 참조하세요.
                                    <example>
                                    자녀가 만원을 내고 5000원짜리 구슬 아이스크림을 사먹었는데 거스름돈을 돌려주지 않았음. 아이가 잔돈을 달라고 했으나 근무자가 대꾸도 안하고 무시했다는 이야기를 들어 활당했으니 개선 바람.
                                    </example>
                                    
                                    응답 형식: 50자 이내로 서술형 문장 1개 또는 2개 이내로 요약된 내용을 표현하세요. 일부 추가 설명은 포함되어도 됩니다. 생성된 답변만 출력하세요.
                                     """, 
                                     model_id),
            '구분': call_bedrock_model(f"""
                                     입력된 텍스트를 다음 VOC 유형 중 하나로 분류해주세요:
                                    - 일반문의
                                    - 제안사항
                                    - 개선요청
                                    - 칭찬격려
                                    - 기타
                                    
                                    분류 기준:
                                    - 일반문의: 서비스나 제품에 대한 단순한 질문이나 정보 요청
                                    - 제안사항: 새로운 아이디어나 서비스 개선을 위한 제안
                                    - 개선요청: 기존 서비스나 제품의 문제점 지적 및 개선 요구
                                    - 칭찬격려: 긍정적인 경험이나 만족에 대한 표현
                                    - 기타: 위 카테고리에 명확히 속하지 않는 내용
                                    
                                    텍스트: <text>{voc_text}</text>
                                    
                                    응답 형식: 위의 5가지 분류 중 가장 적합한 하나만 선택하여 단어 그대로 답변해주세요. 추가 설명이나 주석 없이 분류 단어만 작성해주세요.
                                    
                                    예시 응답:
                                    개선요청
                                     """
                                     , model_id),
            '생성 답변': call_bedrock_model(answer_prompt, model_id),
            '담당 부서': call_bedrock_model(dept_prompt, model_id)
        }
        results.append(result)
    
    # 처리 완료 후 progress bar와 상태 텍스트 제거
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)



def main():
    st.set_page_config(layout="wide")
    logo = Image.open('logo.png')
    st.image(logo, width=200)
    st.title("Samsung C&T Resort - Daily VOC Analysis")

    # session_state 초기화
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    # 초기화 버튼을 제목 바로 아래에 우측 정렬로 배치
    col_reset = st.container()
    with col_reset:
        col1, col2, col3 = st.columns([6, 2, 2])
        with col3:
            if st.button("화면 초기화", use_container_width=True):
                st.session_state.results_df = None
                st.rerun()  # experimental_rerun() 대신 rerun() 사용
    
    # 파일 업로드
    uploaded_file = st.file_uploader("VOC Excel 파일을 업로드하세요", type=['xlsx'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        
        # VOC 분석 버튼
        if st.button("VOC분석"):
            st.write(f"총 {len(df)}건의 VOC를 분석합니다.")
            st.session_state.results_df = process_voc(df)
        
        # 결과가 있을 경우 표시
        if st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            
            # 결과 표시
            st.subheader("분석 결과")
            
            # VOC 유형별 건수 계산
            voc_counts = results_df['구분'].value_counts()
            
            # 유형별 건수 테이블 생성
            st.markdown("### 리조트 일일 VOC 발생 현황")
            
            # 테이블 데이터 준비
            count_data = {
                "구분": ["개선요청", "칭찬격려", "일반문의", "제안사항", "기타"],
                "합계": [
                    voc_counts.get("개선요청", 0),
                    voc_counts.get("칭찬격려", 0),
                    voc_counts.get("일반문의", 0),
                    voc_counts.get("제안사항", 0),
                    voc_counts.get("기타", 0)
                ]
            }
            count_df = pd.DataFrame(count_data)
            
            # 날짜 표시
            current_date = datetime.datetime.now().strftime("%m/%d")
            st.write(f"금일({current_date}) 접수한 손님 VOC는 개선요청 {count_df.loc[count_df['구분']=='개선요청', '합계'].iloc[0]}건, "
                    f"칭찬격려 {count_df.loc[count_df['구분']=='칭찬격려', '합계'].iloc[0]}건, "
                    f"일반문의 {count_df.loc[count_df['구분']=='일반문의', '합계'].iloc[0]}건입니다.")
            
            # 테이블 표시
            st.dataframe(
                count_df,
                column_config={
                    "구분": st.column_config.TextColumn("구분"),
                    "합계": st.column_config.NumberColumn("합계")
                },
                hide_index=True
            )
            
            # VOC 내용 상세 표시
            st.markdown("### VOC 상세 내용")
            
            # 각 VOC 유형별로 내용 표시
            for voc_type in ["개선요청", "칭찬격려", "일반문의", "제안사항", "기타"]:
                if voc_type in voc_counts.index:
                    with st.expander(f"{voc_type} ({voc_counts[voc_type]}건)"):
                        filtered_df = results_df[results_df['구분'] == voc_type]
                        summary_table = pd.DataFrame({
                            'VOC 구분': [voc_type] * len(filtered_df),
                            'VOC 내용 요약': filtered_df['요약'].values
                        })
                        st.dataframe(
                            summary_table,
                            hide_index=True,
                            use_container_width=True
                        )
            
            # CSV 다운로드 버튼
            st.markdown("---")
            csv = results_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="결과(csv) 다운로드",
                data=csv,
                file_name=f"voc_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()