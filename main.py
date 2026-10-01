# Paid API execution blocked by the account owner on 2026-10-01.
# This guard runs before imports, environment loading, document access, or any API call.
raise SystemExit("유료 AI API 사용이 차단되어 ChatPDF 실행을 중지합니다.")

from langchain_classic import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#pdf 로딩
loader = PyPDFLoader("spring.pdf")
pages = loader.load_and_split()
# print(len(pages))
# print(pages[0])
#문서문할
text_splitter = RecursiveCharacterTextSplitter(
  # Set a really small chunk size, just to show.
  chunk_size=100, #각 청크의 최대 길이
  chunk_overlap=20, #인접한 청크사이의 중복영역, 문장이 끊기는 문제 해결 20글자 겹침
  length_function=len,#청크길이 측정하는 함수
  is_separator_regex=False,#단순한 문자열로 해석
)
texts = text_splitter.split_documents(pages)
# print(len(texts))
# print(texts[0])

#임베딩
embeddings_model = OpenAIEmbeddings(api_key=api_key)
#벡터저장소 설정
db = Chroma.from_documents(texts, embeddings_model)
question = "점순이가 주인공에게 느끼는 감정은 무엇이야?"
llm = ChatOpenAI(temperature=0, api_key=api_key)

# Chroma 백터 저장소에 대한 Retriever 인스턴스 생성
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

#사용자 질문에 대한 연관정보 가져온다.
docs = retriever_from_llm.invoke(question)
# print(len(docs)) # 검색기의 실행 결과인 docs의 개수
# print(docs)

#Generate 
     
prompt = hub.pull('rlm/rag-prompt') #Prompt Template

#검색결과 format
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

  # 체인
rag_chain = (
  {'context':retriever_from_llm | format_docs, "question":RunnablePassthrough()}  #입력값 그대로 사용
  | prompt
  | llm
  | StrOutputParser()
)

#실행
result = rag_chain.invoke(question)
print(result)
