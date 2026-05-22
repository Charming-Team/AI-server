import argparse
from datetime import datetime

from app.features.chat.schemas import ChatAnswerRequest, ChatUserContext

DEFAULT_REQUESTED_AT = "2026-05-12T10:30:00+09:00"


def add_chat_request_arguments(
    parser: argparse.ArgumentParser,
    default_question: str,
) -> None:
    parser.add_argument("--question", default=default_question, help="점검 질문")
    parser.add_argument("--role", default="MANUFACTURING_MANAGER", help="사용자 Role")
    parser.add_argument("--user-id", type=int, default=1, help="사용자 ID")
    parser.add_argument("--company-name", default="S-MAP", help="회사명 메타데이터")
    parser.add_argument("--session-id", type=int, default=1, help="세션 ID")
    parser.add_argument("--message-id", type=int, default=1, help="메시지 ID")
    parser.add_argument(
        "--requested-at",
        default=DEFAULT_REQUESTED_AT,
        help="요청 기준 시각. ISO datetime 형식",
    )


def build_chat_answer_request(args: argparse.Namespace) -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=args.session_id,
        messageId=args.message_id,
        user=ChatUserContext(
            userId=args.user_id,
            role=args.role,
            companyName=args.company_name,
            status="ACTIVE",
        ),
        question=args.question,
        requestedAt=datetime.fromisoformat(args.requested_at),
    )
