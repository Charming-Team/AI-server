from argparse import ArgumentParser, Namespace

from scripts import chat_check_common


def test_chat_check_common_adds_request_arguments() -> None:
    parser = ArgumentParser()
    chat_check_common.add_chat_request_arguments(parser, "기본 질문")

    args = parser.parse_args([])

    assert args.question == "기본 질문"
    assert args.role == "MANUFACTURING_MANAGER"
    assert args.user_id == 1
    assert args.company_name == "S-MAP"
    assert args.session_id == 1
    assert args.message_id == 1
    assert args.requested_at == chat_check_common.DEFAULT_REQUESTED_AT


def test_chat_check_common_builds_chat_answer_request() -> None:
    request = chat_check_common.build_chat_answer_request(
        Namespace(
            session_id=10,
            message_id=24,
            user_id=7,
            role=" executive ",
            company_name="S-MAP",
            question="자재 부족 현황 알려줘",
            requested_at="2026-05-12T10:30:00+09:00",
        )
    )

    assert request.session_id == 10
    assert request.message_id == 24
    assert request.user.user_id == 7
    assert request.user.role == "EXECUTIVE"
    assert request.user.status == "ACTIVE"
    assert request.question == "자재 부족 현황 알려줘"
