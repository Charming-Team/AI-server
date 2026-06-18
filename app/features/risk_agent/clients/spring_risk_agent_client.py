from __future__ import annotations

import httpx

from app.features.risk_agent.schemas.evidence import (
    RiskAgentEvidence,
    SpringEnvelope,
)
from app.features.risk_agent.schemas.persist import (
    RiskAgentPersistRequest,
)


class SpringRiskAgentClientError(RuntimeError):
    pass


class SpringRiskAgentClient:
    def __init__(
        self,
        *,
        base_url: str,
        internal_token: str,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.internal_token = internal_token

        self.timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=10.0,
            pool=5.0,
        )

    async def fetch_evidence(
        self,
        prediction_id: int,
        order_id: int,
    ) -> RiskAgentEvidence:
        if not self.internal_token:
            raise SpringRiskAgentClientError(
                "RISK_AGENT_INTERNAL_TOKEN이 설정되지 않았습니다."
            )

        path = (
            f"/internal/risk-agent/evidence/"
            f"{prediction_id}/{order_id}"
        )

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            ) as client:
                response = await client.get(
                    path,
                    headers={
                        "X-Internal-Token": self.internal_token,
                    },
                )

                response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise SpringRiskAgentClientError(
                "Spring Evidence API가 오류를 반환했습니다. "
                f"status={exc.response.status_code}, "
                f"body={exc.response.text[:500]}"
            ) from exc

        except httpx.RequestError as exc:
            raise SpringRiskAgentClientError(
                f"Spring Evidence API 연결에 실패했습니다: {exc}"
            ) from exc

        envelope = SpringEnvelope[RiskAgentEvidence].model_validate(
            response.json()
        )

        if not envelope.success or envelope.data is None:
            raise SpringRiskAgentClientError(
                envelope.message or "Spring Evidence 응답에 data가 없습니다."
            )

        return envelope.data

    async def persist_analysis(
            self,
            request: RiskAgentPersistRequest,
        ) -> None:
            if not self.internal_token:
                raise SpringRiskAgentClientError(
                    "RISK_AGENT_INTERNAL_TOKEN이 설정되지 않았습니다."
                )

            path = "/internal/risk-agent/results"

            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                ) as client:
                    response = await client.post(
                        path,
                        headers={
                            "X-Internal-Token": self.internal_token,
                        },
                        json=request.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                    )

                    response.raise_for_status()

            except httpx.HTTPStatusError as exc:
                raise SpringRiskAgentClientError(
                    "Spring Persist API가 오류를 반환했습니다. "
                    f"status={exc.response.status_code}, "
                    f"body={exc.response.text[:500]}"
                ) from exc

            except httpx.RequestError as exc:
                raise SpringRiskAgentClientError(
                    f"Spring Persist API 연결에 실패했습니다: {exc}"
                ) from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise SpringRiskAgentClientError(
                    "Spring Persist API 응답이 JSON이 아닙니다."
                ) from exc

            if not isinstance(payload, dict):
                raise SpringRiskAgentClientError(
                    "Spring Persist API 응답 형식이 올바르지 않습니다."
                )

            if payload.get("success") is not True:
                raise SpringRiskAgentClientError(
                    str(
                        payload.get("message")
                        or "Spring Persist API 저장에 실패했습니다."
                    )
                )