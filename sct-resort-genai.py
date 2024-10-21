import streamlit as st
import boto3
import json
from PIL import Image
from botocore.client import Config

# Amazon Bedrock 클라이언트 설정
bedrock_runtime = boto3.client('bedrock-runtime')
bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_agent_client = boto3.client("bedrock-agent-runtime", config=bedrock_config)

# Knowledge Base ID 설정
kb_id = "ACSN5MRWK2"  ### <---- 자신의 KB ID로 반드시 변경 필요!!!!!!!!!!!!!!!!!!!!

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

def summarize_input(input_text, model_id):
    prompt = f"""
    다음 텍스트의 핵심 내용을 정확하고 간결하게 요약해주세요: 
    
    <text>
    {input_text}
    </text>
    
    요약 지침:
    1. 반드시 30자 이내로 작성하세요.
    2. 가장 중요한 정보만 포함하세요.
    3. 완전한 문장으로 작성하지 말고, 핵심 키워드나 구문만 사용하세요.
    4. 불필요한 조사, 접속사는 생략하세요.
    5. 원문의 주요 의도나 감정을 반영하세요.
    
    응답 형식: 50자 이내로 서술형 문장으로 요약된 내용을 표현하세요. 일부 추가 설명은 포함되어도 됩니다.
    """
    return call_bedrock_model(prompt, model_id)

def classify_input(input_text, model_id):
    prompt = f"""
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
    
    텍스트: <text>{input_text}</text>
    
    응답 형식: 위의 5가지 분류 중 가장 적합한 하나만 선택하여 단어 그대로 답변해주세요. 추가 설명이나 주석 없이 분류 단어만 작성해주세요.
    
    예시 응답:
    개선요청
    """
    return call_bedrock_model(prompt, model_id)
    

def retrieve(query, kbId, numberOfResults=10):
    return bedrock_agent_client.retrieve(
        retrievalQuery= {
            'text': query
        },
        knowledgeBaseId=kbId,
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': numberOfResults,
                'overrideSearchType': "HYBRID",
            }
        }
    )

def get_contexts(retrievalResults):
    contexts = []
    for retrievedResult in retrievalResults: 
        contexts.append(retrievedResult['content']['text'])
    return contexts

def get_department(input_text, model_id):
    response = retrieve(input_text, kb_id, 10)
    retrievalResults = response['retrievalResults']
    contexts = get_contexts(retrievalResults)

    prompt = f"""
    Human: 당신은 리조트 회사의 고객 서비스를 전문으로 하는 AI 어시스턴트입니다. 새로운 VOC(Voice of Customer)와 유사한 항목을 찾고, 관련 정보와 담당 부서를 제공하는 것이 당신의 임무입니다.
    다음의 맥락 정보를 사용하여 <question> 태그 안에 있는 질문에 답하세요. 맥락에는 이전 VOC 항목들의 "질문내용", "답변내용", "담당 부서" 등의 정보가 포함되어 있습니다.

    <context>
    {contexts}
    </context>

    <question>
    {input_text}
    </question>

    다음 형식으로 응답해 주세요:
    이러한 유형의 문의나 이슈를 처리하는 담당 부서 이름만 답변에 출력하세요.
    추가 설명이나 주석 없이 분류 단어만 작성해주세요.
    
    예시 응답:
    리조트 주토피아운영그룹 동물보전1

    A:"""

    return call_bedrock_model(prompt, model_id)


def gen_feedback(input_text, model_id):
    response = retrieve(input_text, kb_id, 10)
    retrievalResults = response['retrievalResults']
    contexts = get_contexts(retrievalResults)

    prompt = f"""
    Human: 당신은 리조트 회사의 고객 서비스를 전문으로 하는 AI 어시스턴트입니다. 새로운 VOC(Voice of Customer)와 유사한 항목을 찾고, 관련 정보와 담당 부서를 제공하는 것이 당신의 임무입니다.
    다음의 맥락 정보를 사용하여 <question> 태그 안에 있는 질문에 답하세요. 맥락에는 이전 VOC 항목들의 "질문내용", "답변내용", "담당 부서" 등의 정보가 포함되어 있습니다.

    <context>
    {contexts}
    </context>

    <question>
    {input_text}
    </question>

    다음 형식으로 응답해 주세요:
    이러한 유형의 문의나 이슈를 처리하는 답변내용을 참조하세요.
    답변 내용 및 스타일을 참조하여 고객 질의에 대한 답변을 생성하세요. 생성된 답변만 출력하세요.
    
    예시 응답:
    "안전환경그룹고객안전 *** 프로안녕하세요안전그룹 *** 프로 입니다사고자 모 통화병원 치료 흉터 남을 것 같다고 하며 여자아이라 걱정하심보호자 책임도 있다고 공감하시어 당사에서 도의적인 치료비 지원 예정 입니다문제 없도록 종결 하겠습니다감사 합니다"

    A:"""

    return call_bedrock_model(prompt, model_id)


st.set_page_config(layout="wide")
# 로고 이미지 로드
logo = Image.open('logo.png')

# 로고 표시
st.image(logo, width=200)  # width를 조절하여 크기 조정

# 앱 제목 (로고 아래에 표시됨)
st.title("Samsung C&T Resort – GenAI for VOC")

# 모델 선택 옵션 추가
model_option = st.selectbox(
    "모델 선택",
    ("Claude 3 Sonnet", "Claude 3.5 Sonnet"),
    index=1  # 기본값으로 Claude 3.5 Sonnet 선택
)

# 선택된 모델에 따라 model_id 설정
if model_option == "Claude 3 Sonnet":
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
else:
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# 메인 컨텐츠를 위한 컨테이너
main_container = st.container()

with main_container:
    col1, col2 = st.columns([3, 2])

    with col1:
        # VOC 샘플 버튼
        sample1, sample2, sample3 = st.columns(3)
        with sample1:
            if st.button("VOC 샘플 1"):
                st.session_state.voc_input = """어제 아이 생일 기념으로 에버랜드에 방문했어요! 가는길이 멀어서 아이가 지쳐 짜증내고 있던 참에 조류관(?)에 들렸는데요.. 마침 설명해 주시는 시간이었고, 계시던 사육사님께서 재미있게 설명해주시고 퀴즈도 내면서 아이들 참여를 유도해 주시고 아이들 이야기도 잘들어 주셔서 저희 아이도 끝까지 즐겁게 참여할 수 있었습니다.막바지 클로징 인사할때쯤 참여했던 사람들이 그냥 가버리셔서 좀 섭섭(?) 하셨을것 같은데 이렇게라도 응원 해 드리고 싶어서 남겨 봅니다!사육사님 성함은 잊어버려서 아이 사진 찍으면서 찍히셨던 사진올려봅니다!저희 아이 여섯번째 생일이었는데 덕분에 즐거운 시간 보냈습니다 감사합니다:)
                """
        with sample2:
            if st.button("VOC 샘플 2"):
                st.session_state.voc_input = """장애 아이가 있어 부모님까지 모시고 같이 캐리비안 베이를 방문했어요. 들어가자마자 사람이 너무 많고 첫 방문이라 헤매고 있다가 락커룸을 이용하려고 갔더니 1층은 이미 마감이라 계단을 이용해 다른층을 이용해야했는데 아이가 계단 사용이 힘들고 많은 짐들과 어떻게할지 몰라 양해를 구하니 웨이브 락커룸 1시 20분경 '***' 분께서 락커룸 확인 후 1층으로 락커룸 한자리 배려해주셨어요. 저와 아이둘에 할머니까지 넷이라 좁긴했지만 짐과 아이를 안고 계단을 오르내리는 건 너무 위험하고 힘들었기에  그 분께 정말 감사해서 눈물이 날 지경이더라구요. 덕분에 계단 오르내리며 위험하지않게 잘 이용했습니다. 다른 공간을 못찾아 이 곳에 감사인사 전해요.그리고 건의 사항이 있다면 1층 락커룸을 장애인 등 불편한 사람을 위해 몇 칸이라도 배려해줄 수 있으실까요? 엘레베이터도 없어서 위험할 수도 있어서 얘기드려봅니다.
                """
        with sample3:
            if st.button("VOC 샘플 3"):
                st.session_state.voc_input = """안녕하세요 6월 22일 촬영 날짜전에 잠시 규정에 대해서 알고싶어서 연락드렸습니다 개인 미러리스로 혼자 촬영하려하구요 개인 스냅 촬영을 하는 브이로그를 촬영하려 합니다 주변인물들은 전부 모자이크를 할것이구요 확정된 인원만 나오는걸로 하여 영상 제작을 하려합니다 에버랜드에 정확한 규정이 어디있는지 몰라 연락드렸습니다 영상은 추후에 유튜브에 업로드 될 예정이오나 아직 채널도 없고 영상 시작을 에버랜드에서 하고자 연락드린겁니다 감사합니다스튜디오 " 도달 " 입니다감사합니다 좋은 하루 보내세요
                """
        
        # 조건부로 초기값 설정
        if "voc_input" not in st.session_state:
            st.session_state.voc_input = ""
        
        # VOC 입력 텍스트 영역
        voc_input = st.text_area("여기에 VOC를 입력해주세요.", height=200, key="voc_input")
        
        # 체크박스와 생성 버튼
        col_check, col_generate = st.columns([3, 1])
        with col_check:
            summary = st.checkbox("요약", key="summary")
            classification = st.checkbox("분류", key="classification")
            feedback = st.checkbox("답변", key="feedback")
            department = st.checkbox("담당 부서", key="department")
        with col_generate:
            generate = st.button("생성형 AI 생성", type="primary")

    with col2:
        st.subheader(f"Amazon Bedrock - {model_option} 답변 결과입니다.")
        
        if generate and voc_input and (summary or classification or feedback or department):
            if summary:
                st.text_area("요약", summarize_input(voc_input, model_id), height=100)
            if classification:
                st.text_area("분류 (일반/제안/개선/칭찬/기타)", classify_input(voc_input, model_id), height=100)
            if feedback:
                st.text_area("답변", gen_feedback(voc_input, model_id), height=200)
            if department:
                st.text_area("담당 부서", get_department(voc_input, model_id), height=100)
        elif generate:
            st.warning("텍스트를 입력하고 최소 하나의 옵션을 선택해주세요.")

# 4월 요약 보고서 버튼 및 내용 표시
st.markdown("---")  # 구분선 추가
if st.button("2024년 4월 VOC 요약 보고서"):
    try:
        with open("./output/2024-04_detailed_summary_report.md", "r", encoding="utf-8") as file:
            report_content = file.read()
        st.markdown(report_content)
    except FileNotFoundError:
        st.error("보고서 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {str(e)}")