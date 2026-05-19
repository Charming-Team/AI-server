import json
from argparse import Namespace
from io import StringIO

import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode
from scripts import validate_document_payload


def _build_payload(**overrides):
    payload = {
        "documentId": "report-202605",
        "documentType": "REPORT",
        "title": "2026년 5월 생산 리스크 보고서",
        "content": "자재 부족과 라인 병목이 주요 리스크입니다.",
        "url": "/reports/20",
        "allowedRoles": ["EXECUTIVE", "MANUFACTURING_MANAGER"],
        "companyName": "S-MAP",
        "intentTags": ["REPORT_LOOKUP"],
    }
    payload.update(overrides)
    return payload


def _write_payload(tmp_path, payload, file_name: str = "document-payload.json"):
    input_path = tmp_path / file_name
    input_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return input_path


def test_validate_document_payload_script_builds_settings_from_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env.document"
    env_file.write_text(
        "\n".join(
            [
                "DOCUMENT_CONTENT_MAX_CHARS=2000",
                "DOCUMENT_MAX_CHUNKS=3",
                "DOCUMENT_CHUNK_SIZE=50",
                "DOCUMENT_CHUNK_OVERLAP=5",
            ]
        ),
        encoding="utf-8",
    )

    settings = validate_document_payload.build_settings(
        Namespace(env_file=str(env_file), json=False, input="document-payload.json")
    )

    assert settings.document_content_max_chars == 2000
    assert settings.document_max_chunks == 3
    assert settings.document_chunk_size == 50
    assert settings.document_chunk_overlap == 5


def test_validate_document_payload_script_validates_payload_without_network(tmp_path) -> None:
    input_path = _write_payload(tmp_path, _build_payload())
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        ["--input", str(input_path)],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "status=VALID" in stdout.getvalue()
    assert "documentId=report-202605" in stdout.getvalue()
    assert "documentType=REPORT" in stdout.getvalue()
    assert "chunkCount=1" in stdout.getvalue()
    assert "networkChecked=False" in stdout.getvalue()
    assert "estimatedEmbeddingRequestCount=0" in stdout.getvalue()
    assert "estimatedQdrantUpsertPointCount=0" in stdout.getvalue()
    assert "warning=임베딩 기능이 비활성화되어 실제 문서 저장은 생략됩니다." in (
        stdout.getvalue()
    )


def test_validate_document_payload_script_validates_multiple_inputs(tmp_path) -> None:
    first_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-a"),
        "report-a.json",
    )
    second_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-b"),
        "report-b.json",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(first_input),
            "--input",
            str(second_input),
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "status=VALID" in stdout.getvalue()
    assert "inputCount=2" in stdout.getvalue()
    assert "validCount=2" in stdout.getvalue()
    assert "invalidCount=0" in stdout.getvalue()
    assert "documentId=report-202605-a" in stdout.getvalue()
    assert "documentId=report-202605-b" in stdout.getvalue()


def test_validate_document_payload_script_validates_input_directory(tmp_path) -> None:
    input_dir = tmp_path / "payloads"
    input_dir.mkdir()
    _write_payload(
        input_dir,
        _build_payload(documentId="report-202605-a"),
        "report-a.json",
    )
    _write_payload(
        input_dir,
        _build_payload(documentId="report-202605-b"),
        "report-b.json",
    )
    (input_dir / "README.txt").write_text("not a payload", encoding="utf-8")
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        ["--input-dir", str(input_dir)],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "inputCount=2" in stdout.getvalue()
    assert "validCount=2" in stdout.getvalue()
    assert "documentId=report-202605-a" in stdout.getvalue()
    assert "documentId=report-202605-b" in stdout.getvalue()
    assert "README.txt" not in stdout.getvalue()


def test_validate_document_payload_script_combines_input_and_input_directory(
    tmp_path,
) -> None:
    direct_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-a"),
        "report-a.json",
    )
    input_dir = tmp_path / "payloads"
    input_dir.mkdir()
    _write_payload(
        input_dir,
        _build_payload(documentId="report-202605-b"),
        "report-b.json",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(direct_input),
            "--input-dir",
            str(input_dir),
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "inputCount=2" in stdout.getvalue()
    assert "documentId=report-202605-a" in stdout.getvalue()
    assert "documentId=report-202605-b" in stdout.getvalue()


def test_validate_document_payload_script_rejects_missing_input_directory(
    tmp_path,
) -> None:
    stderr = StringIO()

    exit_code = validate_document_payload.main(
        ["--input-dir", str(tmp_path / "missing")],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "문서 payload 디렉터리를 찾을 수 없습니다." in stderr.getvalue()
    assert "CHAT_DOCUMENT_002" in stderr.getvalue()


def test_validate_document_payload_script_rejects_empty_input_directory(
    tmp_path,
) -> None:
    input_dir = tmp_path / "payloads"
    input_dir.mkdir()
    (input_dir / "README.txt").write_text("not a payload", encoding="utf-8")
    stderr = StringIO()

    exit_code = validate_document_payload.main(
        ["--input-dir", str(input_dir)],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "문서 payload 디렉터리에 JSON 파일이 없습니다." in stderr.getvalue()
    assert "CHAT_DOCUMENT_002" in stderr.getvalue()


def test_validate_document_payload_script_requires_input_or_input_directory() -> None:
    stderr = StringIO()

    exit_code = validate_document_payload.main([], stderr=stderr)

    assert exit_code == 1
    assert "검증할 문서 payload 파일 또는 디렉터리가 필요합니다." in stderr.getvalue()
    assert "CHAT_DOCUMENT_002" in stderr.getvalue()


def test_validate_document_payload_script_reports_multiple_input_errors(
    tmp_path,
) -> None:
    valid_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-a"),
        "report-a.json",
    )
    invalid_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-b", documentType="PROCESS"),
        "report-b.json",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(valid_input),
            "--input",
            str(invalid_input),
        ],
        stdout=stdout,
    )

    assert exit_code == 1
    assert "status=INVALID" in stdout.getvalue()
    assert "inputCount=2" in stdout.getvalue()
    assert "validCount=1" in stdout.getvalue()
    assert "invalidCount=1" in stdout.getvalue()
    assert "code=CHAT_DOCUMENT_001" in stdout.getvalue()


def test_validate_document_payload_script_rejects_duplicate_document_ids(
    tmp_path,
) -> None:
    first_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-duplicate"),
        "report-a.json",
    )
    second_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-duplicate"),
        "report-b.json",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(first_input),
            "--input",
            str(second_input),
        ],
        stdout=stdout,
    )

    assert exit_code == 1
    assert "status=INVALID" in stdout.getvalue()
    assert "validCount=0" in stdout.getvalue()
    assert "invalidCount=2" in stdout.getvalue()
    assert "code=CHAT_DOCUMENT_002" in stdout.getvalue()
    assert "documentId가 중복되었습니다" in stdout.getvalue()


def test_validate_document_payload_script_rejects_duplicate_document_ids_in_json(
    tmp_path,
) -> None:
    first_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-duplicate"),
        "report-a.json",
    )
    second_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-duplicate"),
        "report-b.json",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(first_input),
            "--input",
            str(second_input),
            "--json",
        ],
        stdout=stdout,
    )

    assert exit_code == 1
    assert '"invalidCount": 2' in stdout.getvalue()
    assert '"code": "CHAT_DOCUMENT_002"' in stdout.getvalue()
    assert "documentId가 중복되었습니다" in stdout.getvalue()


def test_validate_document_payload_script_prints_json_batch_without_content(
    tmp_path,
) -> None:
    content = "라인 병목 상세 본문입니다."
    first_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-a", content=content),
        "report-a.json",
    )
    second_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-b"),
        "report-b.json",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(first_input),
            "--input",
            str(second_input),
            "--json",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert '"inputCount": 2' in stdout.getvalue()
    assert '"validCount": 2' in stdout.getvalue()
    assert '"results": [' in stdout.getvalue()
    assert content not in stdout.getvalue()


def test_validate_document_payload_script_returns_two_for_batch_warning_strict(
    tmp_path,
) -> None:
    first_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-a"),
        "report-a.json",
    )
    second_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-b"),
        "report-b.json",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(first_input),
            "--input",
            str(second_input),
            "--fail-on-warning",
        ],
        stdout=stdout,
    )

    assert exit_code == 2
    assert "warningCount=2" in stdout.getvalue()


def test_validate_document_payload_script_batch_invalid_precedes_warning_strict(
    tmp_path,
) -> None:
    valid_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-a"),
        "report-a.json",
    )
    invalid_input = _write_payload(
        tmp_path,
        _build_payload(documentId="report-202605-b", documentType="PROCESS"),
        "report-b.json",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(valid_input),
            "--input",
            str(invalid_input),
            "--fail-on-warning",
        ],
        stdout=stdout,
    )

    assert exit_code == 1
    assert "invalidCount=1" in stdout.getvalue()


def test_validate_document_payload_script_returns_two_when_warning_is_strict(
    tmp_path,
) -> None:
    input_path = _write_payload(tmp_path, _build_payload())
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        ["--input", str(input_path), "--fail-on-warning"],
        stdout=stdout,
    )

    assert exit_code == 2
    assert "status=VALID" in stdout.getvalue()
    assert "warning=임베딩 기능이 비활성화되어 실제 문서 저장은 생략됩니다." in (
        stdout.getvalue()
    )


def test_validate_document_payload_script_returns_zero_when_strict_without_warning(
    tmp_path,
) -> None:
    input_path = _write_payload(tmp_path, _build_payload())
    env_file = tmp_path / ".env.document"
    env_file.write_text("EMBEDDING_ENABLED=true", encoding="utf-8")
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(input_path),
            "--env-file",
            str(env_file),
            "--fail-on-warning",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "status=VALID" in stdout.getvalue()
    assert "warning=" not in stdout.getvalue()


def test_validate_document_payload_script_returns_two_for_json_warning(
    tmp_path,
) -> None:
    input_path = _write_payload(tmp_path, _build_payload())
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(input_path),
            "--json",
            "--fail-on-warning",
        ],
        stdout=stdout,
    )

    assert exit_code == 2
    assert '"status": "VALID"' in stdout.getvalue()
    assert '"warnings": [' in stdout.getvalue()


def test_validate_document_payload_script_prints_json_without_document_content(
    tmp_path,
) -> None:
    secret_content = "라인 병목 리포트 본문 원문입니다."
    input_path = _write_payload(tmp_path, _build_payload(content=secret_content))
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        ["--input", str(input_path), "--json"],
        stdout=stdout,
    )

    assert exit_code == 0
    assert '"status": "VALID"' in stdout.getvalue()
    assert '"documentId": "report-202605"' in stdout.getvalue()
    assert secret_content not in stdout.getvalue()


def test_validate_document_payload_script_prints_safe_chunk_metadata(
    tmp_path,
) -> None:
    content = "A" * 30 + "\n" + "B" * 30
    input_path = _write_payload(tmp_path, _build_payload(content=content))
    env_file = tmp_path / ".env.document"
    env_file.write_text(
        "\n".join(
            [
                "DOCUMENT_CHUNK_SIZE=35",
                "DOCUMENT_CHUNK_OVERLAP=5",
            ]
        ),
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(input_path),
            "--env-file",
            str(env_file),
            "--include-chunks",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "chunk=chunk-0001 charCount=30" in stdout.getvalue()
    assert "chunk=chunk-0002 charCount=30" in stdout.getvalue()
    assert "A" * 30 not in stdout.getvalue()
    assert "B" * 30 not in stdout.getvalue()


def test_validate_document_payload_script_prints_json_chunk_metadata_without_content(
    tmp_path,
) -> None:
    content = "라인 병목 상세 본문입니다."
    input_path = _write_payload(tmp_path, _build_payload(content=content))
    stdout = StringIO()

    exit_code = validate_document_payload.main(
        [
            "--input",
            str(input_path),
            "--include-chunks",
            "--json",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert '"chunks": [' in stdout.getvalue()
    assert '"chunkId": "chunk-0001"' in stdout.getvalue()
    assert '"charCount":' in stdout.getvalue()
    assert content not in stdout.getvalue()


def test_validate_document_payload_result_omits_chunks_by_default() -> None:
    result = validate_document_payload.validate_document_payload(
        _build_payload(),
        Settings(),
    )

    assert "chunks" not in result


def test_validate_document_payload_result_estimates_embedding_and_qdrant_work() -> None:
    result = validate_document_payload.validate_document_payload(
        _build_payload(content="A" * 30 + "\n" + "B" * 30),
        Settings(
            embedding_enabled=True,
            document_chunk_size=35,
            document_chunk_overlap=5,
        ),
    )

    assert result["chunkCount"] == 2
    assert result["contentCharCount"] == 61
    assert result["embeddingEnabled"] is True
    assert result["embeddingInputCount"] == 2
    assert result["uniqueEmbeddingInputCount"] == 2
    assert result["estimatedEmbeddingRequestCount"] == 1
    assert result["estimatedQdrantUpsertPointCount"] == 2
    assert result["warnings"] == []


def test_validate_document_payload_result_warns_for_duplicate_chunks() -> None:
    result = validate_document_payload.validate_document_payload(
        _build_payload(content="A" * 30 + "\n" + "A" * 30),
        Settings(
            embedding_enabled=True,
            document_chunk_size=35,
            document_chunk_overlap=5,
        ),
    )

    assert result["embeddingInputCount"] == 2
    assert result["uniqueEmbeddingInputCount"] == 1
    assert "중복 청크가 있어 임베딩 요청 입력은 중복 제거 후 계산됩니다." in (
        result["warnings"]
    )


def test_validate_document_payload_result_warns_when_content_is_near_limit() -> None:
    result = validate_document_payload.validate_document_payload(
        _build_payload(content="A" * 800),
        Settings(
            embedding_enabled=True,
            document_content_max_chars=1000,
            document_chunk_size=1000,
            document_max_chunks=10,
        ),
    )

    assert "문서 본문 길이가 설정 한도에 근접했습니다." in result["warnings"]


def test_validate_document_payload_result_warns_when_chunk_count_is_near_limit() -> None:
    result = validate_document_payload.validate_document_payload(
        _build_payload(content="A\nB\nC\nD"),
        Settings(
            embedding_enabled=True,
            document_chunk_size=1,
            document_chunk_overlap=0,
            document_max_chunks=5,
        ),
    )

    assert result["chunkCount"] == 4
    assert "문서 청크 수가 설정 한도에 근접했습니다." in result["warnings"]


def test_validate_document_payload_resolves_exit_code_from_warning_policy() -> None:
    result = {"warnings": ["warning"]}

    assert validate_document_payload.resolve_exit_code(result, fail_on_warning=False) == 0
    assert validate_document_payload.resolve_exit_code(result, fail_on_warning=True) == 2
    assert validate_document_payload.resolve_exit_code({"warnings": []}, True) == 0


def test_validate_document_payload_marks_duplicate_document_ids_invalid() -> None:
    results = [
        {
            "status": "VALID",
            "documentId": "report-202605-duplicate",
            "warnings": ["warning"],
        },
        {
            "status": "VALID",
            "documentId": "report-202605-duplicate",
            "warnings": [],
        },
        {
            "status": "VALID",
            "documentId": "report-202605-unique",
            "warnings": [],
        },
    ]

    validate_document_payload.apply_duplicate_document_id_errors(results)

    assert results[0]["status"] == "INVALID"
    assert results[1]["status"] == "INVALID"
    assert results[2]["status"] == "VALID"
    assert results[0]["warnings"] == []
    assert results[0]["error"]["code"] == "CHAT_DOCUMENT_002"


def test_validate_document_payload_script_rejects_operator_financial_content(
    tmp_path,
) -> None:
    input_path = _write_payload(
        tmp_path,
        _build_payload(
            content="계약 금액과 패널티 정보가 포함된 보고서입니다.",
            allowedRoles=["OPERATOR", "EXECUTIVE"],
            intentTags=["DELIVERY_RISK"],
        ),
    )
    stderr = StringIO()

    exit_code = validate_document_payload.main(
        ["--input", str(input_path)],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "문서 payload 검증 실패" in stderr.getvalue()
    assert "CHAT_DOCUMENT_002" in stderr.getvalue()


def test_validate_document_payload_script_rejects_unauthorized_company_info_indexer(
    tmp_path,
) -> None:
    input_path = _write_payload(
        tmp_path,
        _build_payload(
            documentType="COMPANY_INFO",
            requestedByRole="OPERATOR",
            intentTags=["PRODUCTION_PLAN"],
        ),
    )
    stderr = StringIO()

    exit_code = validate_document_payload.main(
        ["--input", str(input_path), "--json"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert '"code": "CHAT_SECURITY_004"' in stderr.getvalue()
    assert "회사정보 문서 인덱싱은 ADMIN 또는 MANUFACTURING_MANAGER만" in stderr.getvalue()


def test_validate_document_payload_script_rejects_invalid_json(tmp_path) -> None:
    input_path = tmp_path / "document-payload.json"
    input_path.write_text("{", encoding="utf-8")
    stderr = StringIO()

    exit_code = validate_document_payload.main(
        ["--input", str(input_path)],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "문서 payload JSON 형식이 올바르지 않습니다." in stderr.getvalue()
    assert "CHAT_DOCUMENT_002" in stderr.getvalue()


def test_validate_document_payload_raises_when_chunk_count_exceeds_limit() -> None:
    payload = _build_payload(content="A\nB\nC")

    with pytest.raises(ChatServiceError) as exc_info:
        validate_document_payload.validate_document_payload(
            payload,
            Settings(document_chunk_size=1, document_chunk_overlap=0, document_max_chunks=1),
        )

    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_002
