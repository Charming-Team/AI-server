# S-MAP AI Server

FastAPI backend for S-MAP services.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If you prefer editable package installation:

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn app.main:app --reload
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/ai/api/v1/health

## 리포지토리 구조

```text
.
├── app/
│   ├── api/                 # FastAPI 라우터 조립과 v1 API 진입점
│   ├── core/                # 환경변수, Settings, 공통 설정
│   ├── features/
│   │   └── chat/            # RAG 챗봇 핵심 기능
│   │       ├── router.py    # 챗봇 API, 내부 토큰 검증
│   │       ├── service.py   # 질문 처리 전체 흐름 조립
│   │       ├── *_policy.py  # Role, 보안, 문서 접근, URL 정책
│   │       ├── *_client.py  # Qdrant, Embedding, LLM, RDB View 연동
│   │       └── *_service.py # 추천 질문, 문서 인덱싱, 답변 생성 등 기능 단위 서비스
│   ├── repositories/        # 공통 저장소 계층이 필요할 때 사용
│   ├── schemas/             # 공통 API 응답 모델
│   └── services/            # 공통 서비스 계층이 필요할 때 사용
├── scripts/                 # 로컬/운영 점검 스크립트
│   └── postgres/            # 챗봇용 PostgreSQL Read-only View SQL
├── tests/                   # 단위 테스트와 시나리오 테스트
├── deploy/                  # Kubernetes 배포 보조 파일
├── .github/workflows/       # CI/CD 워크플로우
├── Dockerfile               # FastAPI 이미지 빌드 설정
├── pyproject.toml           # 패키지, pytest, ruff 설정
└── README.md
```

### 챗봇 핵심 흐름

```text
Spring 내부 호출
→ app/features/chat/router.py
→ ChatService
→ 질문 검증 / 의도 분류 / Role 검사
→ RDB Evidence View 조회 + Qdrant 문서 검색
→ Grounded Prompt 생성
→ LLM 답변 생성
→ 답변 후처리 / 출처 구성
→ Spring으로 응답 반환
```

## Test

```bash
pytest
```
