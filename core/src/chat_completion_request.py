from pydantic import BaseModel
from chat_message import ChatMessage


class ChatCompletionRequest(BaseModel):
    model: str
    stream: bool = False
    messages: list[ChatMessage]
