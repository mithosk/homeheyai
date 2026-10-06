import time
import uuid
from typing import Any
from fastapi import APIRouter, FastAPI, Request
from starlette.concurrency import run_in_threadpool
from chat_completion_request import ChatCompletionRequest


class OpenAIRouter:
    def __init__(self):
        self._router = APIRouter()
        self._router.add_api_route("/models", self._models, methods=["GET"])
        self._router.add_api_route(
            "/chat/completions", self._completions, methods=["POST"]
        )

    def _models(self, request: Request) -> dict[str, Any]:
        return {
            "data": [
                {
                    "created": int(time.time()),
                    "id": agent.name,
                    "object": "model",
                    "owned_by": "homeheyai",
                }
                for agent in request.app.state.agents
            ],
            "object": "list",
        }

    async def _completions(
        self, request: Request, chat_completion_request: ChatCompletionRequest
    ) -> dict[str, Any]:
        prompt = "\n".join(
            f"{message.role}: {message.content}"
            for message in chat_completion_request.messages
        )

        agent = next(
            (
                agent
                for agent in request.app.state.agents
                if agent.name == chat_completion_request.model
            ),
            None,
        )

        content = await run_in_threadpool(agent.respond, prompt=prompt)

        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": content, "role": "assistant"},
                }
            ],
            "created": int(time.time()),
            "id": str(uuid.uuid4()),
            "model": chat_completion_request.model,
            "object": "chat.completion",
            "usage": None,
        }

    def mount(self, app: FastAPI) -> None:
        app.include_router(self._router)
