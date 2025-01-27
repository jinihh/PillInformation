# LangChain 라이브러리의 최신 버전에서 사용된 API가 변경 -> 최신 패키지로 업데이트
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.chat_models import ChatOllama
from langchain_ollama import ChatOllama
#from langchain.embeddings import FastEmbedEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 추가된 부분
from langchain.docstore.document import Document  # Document 클래스 추가

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, text_file_path: str):
        # 모델 초기화 (ChatOllama 모델 사용)
        self.model = ChatOllama(model="mistral", server_url="http://localhost:11434")
        # 텍스트를 일정 크기로 나누기 위한 설정 (청크 크기 1024, 오버랩 100)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # 질문을 처리할 템플릿 설정 (간결하고 짧은 답변 요구)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s>
            [INST] Question: {question} Context: {context} Answer: [/INST]
            """
        )
        # 텍스트 파일에서 텍스트를 읽어와서 인덱싱
        self.ingest(text_file_path)

    def ingest(self, text_file_path: str):
        # 텍스트 파일에서 텍스트 읽기
        with open(text_file_path, "r", encoding="utf-8") as file:
            extracted_text = file.read()
        
        # 텍스트를 문서 청크로 나누기
        chunks = self.text_splitter.split_text(extracted_text)
        # 청크를 Document 객체로 변환
        documents = [Document(page_content=chunk) for chunk in chunks]
        # 복잡한 메타데이터 필터링
        documents = filter_complex_metadata(documents)
        # 벡터 스토어 생성 (임베딩 사용)
        vector_store = Chroma.from_documents(documents=documents, embedding=FastEmbedEmbeddings())
        # 리트리버 설정 (유사도 기준으로 문서 검색)
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,  # 상위 3개의 결과 반환
                "score_threshold": 0.5,  # 유사도 점수 임계값
            },
        )
        # 질문-응답 체인 설정 (리트리버, 프롬프트, 모델, 출력 파서 사용)
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()} | self.prompt | self.model | StrOutputParser())

    def ask(self, query: str):
        # 체인이 설정되지 않은 경우 오류 메시지 반환
        if not self.chain:
            return "Please, add a text document first."
        # 체인을 통해 질문에 대한 답변 반환
        return self.chain.invoke(query)

    def clear(self):
        # 벡터 스토어, 리트리버, 체인 초기화
        self.vector_store = None
        self.retriever = None
        self.chain = None

# 사용 예시
if __name__ == "__main__":
    # 추출된 텍스트 파일을 사용하여 ChatPDF 초기화
    text_file_path = "data/extracted_text.txt"
    chat_pdf = ChatPDF(text_file_path)
    # 질문 입력
    question = "What is the main use of the drug mentioned?"
    # 질문에 대한 답변 출력
    print(chat_pdf.ask(question))
