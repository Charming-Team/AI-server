# S-MAP AI Server

S-MAP 서비스의 AI 기능을 담당하는 FastAPI 서버입니다. Spring Back-end가 내부 API로 호출하는 챗봇, 생산 계획 최적화, 지연 예측, 리포트 생성, Risk Agent 기능을 제공합니다.

## 주요 기능

- RAG 챗봇 답변 생성, 추천 질문 생성, 내부 문서 인덱싱/삭제
- 생산 운영 리포트 및 경영 리포트 생성
- 주문별 지연 확률 예측과 기존 지연 예측 API
- CP-SAT 기반 생산 계획 조정 및 대시보드 분석 생성
- 지연 원인 분석을 위한 Risk Agent workflow
- 헬스 체크, readiness, 로컬/운영 smoke check 스크립트

## 로컬 실행

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

`requirements.txt`에 Windows에서 설치되지 않는 `uvloop`이 포함될 수 있습니다. 로컬 Windows 검증이 필요하면 `uvloop`을 제외한 임시 requirements로 설치하거나, `pip install -e ".[dev]"`로 최소 개발 의존성을 설치해 테스트 범위를 좁혀 실행하세요.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

개발용 패키지 설치:

```bash
pip install -e ".[dev]"
```

## 접속 URL

기본 `API_V1_PREFIX`는 `/api/v1`입니다.

- Root: <http://127.0.0.1:8000/>
- API docs: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>
- Health: <http://127.0.0.1:8000/api/v1/health>
- Readiness: <http://127.0.0.1:8000/api/v1/health/ready>

외부 `/ai` prefix는 인프라에서 붙이는 값을 기준으로 운용합니다. 애플리케이션 내부 기본 prefix는 `.env`의 `API_V1_PREFIX`로 관리합니다.

## 주요 API

모든 경로는 기본적으로 `/api/v1` 아래에 등록됩니다.

| 기능 | 메서드 / 경로 | 설명 |
| --- | --- | --- |
| Health | `GET /health` | 라우트 가용성 확인 |
| Readiness | `GET /health/ready` | RDB, Qdrant, Embedding, LLM 등 런타임 설정 상태 확인 |
| Chat | `POST /chat/answer` | RDB Evidence와 Qdrant 검색을 결합한 챗봇 답변 생성 |
| Chat | `POST /chat/recommendations` | 사용자 Role/상태 기반 추천 질문 생성 |
| Document | `POST /chat/internal/documents/index` | 내부 문서 payload를 Qdrant에 인덱싱 |
| Document | `POST /chat/internal/documents/delete` | 내부 문서 인덱스 삭제 |
| Report | `POST /reports/generate` | 생산 운영 리포트 생성 |
| Report | `GET /reports` | 생성된 리포트 목록 조회 |
| Report | `GET /reports/{report_id}` | 생성된 리포트 상세 조회 |
| Business Report | `POST /business-reports/generate` | 생산 리포트를 경영 리포트로 변환 |
| Delay Prediction | `POST /delay-prediction/predict` | 기존 지연 예측 API |
| Delay Probability | `POST /delay-probability/predict` | `orderId` 기준 주문별 지연 확률, 위험도, 원인 상세 반환 |
| Planning | `POST /planning` | 생산 계획 조정, CP-SAT 최적화, 시뮬레이션 분석 |
| Risk Agent | `POST /risk-agent/execute` | 지연 원인 분석 workflow 전체 실행 |

Spring 내부 호출용 API는 `X-Internal-Token` 헤더를 사용합니다. 기능별 토큰은 `.env.example`의 `CHAT_*_INTERNAL_TOKEN`, `DOCUMENT_INDEX_INTERNAL_TOKEN`, `RISK_AGENT_INTERNAL_TOKEN` 값을 참고하세요.

## 환경변수

`.env.example`을 복사한 뒤 필요한 값만 채웁니다.

| 영역 | 주요 변수 |
| --- | --- |
| 공통 | `APP_NAME`, `ENVIRONMENT`, `API_V1_PREFIX`, `CORS_ORIGINS` |
| Chat 보안 | `CHAT_ANSWER_INTERNAL_TOKEN`, `CHAT_RECOMMENDATION_INTERNAL_TOKEN` |
| Spring Evidence | `EVIDENCE_LOOKUP_ENABLED`, `EVIDENCE_LOOKUP_BASE_URL`, `EVIDENCE_LOOKUP_INTERNAL_TOKEN` |
| RDB Evidence | `RDB_EVIDENCE_ENABLED`, `RDB_EVIDENCE_DSN` |
| Report / Planning DB | `REPORT_RDB_DSN`, `PLANNING_RDB_DSN` |
| Qdrant | `QDRANT_SEARCH_ENABLED`, `QDRANT_URL`, `QDRANT_COLLECTION` |
| Document Index | `DOCUMENT_INDEX_INTERNAL_TOKEN`, `DOCUMENT_CHUNK_SIZE`, `DOCUMENT_CHUNK_OVERLAP` |
| Embedding | `EMBEDDING_ENABLED`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` |
| LLM | `LLM_ENABLED`, `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_ALLOWED_MODELS` |
| Risk Agent | `RISK_AGENT_INTERNAL_TOKEN`, `RISK_AGENT_SPRING_BASE_URL`, `RISK_AGENT_MAX_RETRIES` |

## 검증 명령

```powershell
python -m ruff check .
python -m pytest
```

특정 기능만 빠르게 볼 때:

```powershell
python -m pytest tests\test_health.py tests\test_fastapi_api_contract.py
python -m pytest tests\test_chat.py
python -m pytest tests\test_production_planning_api.py
```

챗봇/RAG 런타임 점검 스크립트:

```powershell
python -m scripts.check_chat_readiness
python -m scripts.check_chat_runtime --preset full --json
python -m scripts.check_rag_end_to_end --help
```

## 리포지토리 구조

```text
.
├── app/
│   ├── api/                 # FastAPI 라우터 조립과 v1 API 진입점
│   ├── core/                # 환경변수, Settings, 공통 설정
│   ├── features/
│   │   ├── chat/            # RAG 챗봇, 추천 질문, 문서 인덱싱
│   │   ├── business_report/ # 경영 리포트 생성
│   │   ├── delay_probability/ # 주문별 지연 확률 예측
│   │   ├── production_planning/ # 생산 계획 최적화와 시뮬레이션
│   │   ├── report/          # 생산 운영 리포트
│   │   └── risk_agent/      # 지연 원인 분석 workflow
│   ├── repositories/        # 공통 저장소 계층
│   ├── schemas/             # 공통 API 응답 모델
│   └── services/            # 공통 서비스 계층
├── scripts/                 # 로컬/운영 점검 및 데이터 준비 스크립트
│   └── postgres/            # 챗봇용 PostgreSQL read-only view SQL
├── tests/                   # 단위 테스트와 API 계약 테스트
├── deploy/                  # Kubernetes 배포 보조 파일
├── .github/workflows/       # CI/CD workflow
├── Dockerfile               # FastAPI 이미지 빌드 설정
├── pyproject.toml           # 패키지, pytest, ruff 설정
└── README.md
```

## 기능별 핵심 흐름

### 챗봇 답변

```text
Spring 내부 호출
-> app/features/chat/router.py
-> ChatService
-> 질문 검증 / 의도 분류 / Role 검사
-> RDB Evidence View 조회 + Qdrant 문서 검색
-> Grounded Prompt 생성
-> LLM 답변 생성
-> 답변 후처리 / 출처 구성
-> Spring으로 응답 반환
```

### 내부 문서 인덱싱

```text
Spring 내부 호출
-> app/features/chat/router.py
-> DocumentIndexService
-> 문서 payload 검증 / chunk 분할
-> Embedding 생성
-> Qdrant collection 저장 또는 삭제
-> 처리 결과 반환
```

### 생산 운영 리포트

```text
Spring 또는 관리자 화면 호출
-> app/api/v1/routes/report.py
-> ReportGenerationService
-> read-only 운영 DB view 조회
-> 리포트 섹션 구성
-> LLM 요약/분석 생성
-> 리포트 저장 결과 또는 조회 결과 반환
```

### 경영 리포트

```text
Spring 또는 관리자 화면 호출
-> app/api/v1/routes/business_report.py
-> BusinessReportGenerationService
-> 기존 생산 리포트 조회
-> 경영 관점 요약/인사이트 구성
-> LLM 기반 경영 리포트 생성
-> 변환 결과 반환
```

### 주문별 지연 확률 예측

```text
Spring 내부 호출
-> app/api/v1/routes/delay_probability.py
-> DelayProbabilityPredictionService
-> orderId 기준 inference view 조회
-> XGBoost artifact 로드 및 예측
-> riskLevel / delayProbability / causeDetail 구성
-> Spring으로 응답 반환
```

### 생산 계획 조정

```text
Spring 또는 프론트엔드 호출
-> app/api/v1/routes/planning.py
-> Production Planning workflow
-> editOrders 고정 / addOrders 신규 배치
-> CP-SAT 최적화 변형 생성
-> Monte Carlo 시뮬레이션 평가
-> planning_response + simulation_response 반환
```

### Risk Agent

```text
Spring 내부 호출
-> app/features/risk_agent/router.py
-> RiskAgentWorkflowController
-> Spring Evidence API로 컨텍스트 조회
-> 자재 / 수율 / 설비 / 라인 / 납기 Analyzer 실행
-> 원인 우선순위 산정 및 LLM 설명 생성
-> 검증 후 Spring 저장 API 호출
```
