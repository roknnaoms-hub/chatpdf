# chatPDF

PDF 문서를 업로드하고 질문하면, 문서 내용을 기반으로 답변을 생성하는 Streamlit 기반 chatPDF 프로젝트임.

## 주요 기능
- PDF 파일 업로드
- 문서 청크 분할
- OpenAI 임베딩 생성
- Chroma 벡터DB 저장
- MultiQueryRetriever 기반 문서 검색
- RAG 기반 질문응답

## 실행 환경
- Python 3.x
- Streamlit
- LangChain
- OpenAI API
- ChromaDB

## 실행 방법

### 1. 패키지 설치
```bash
pip install -r requirements.txt