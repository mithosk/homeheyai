import os
import json
import uvicorn
from agent import Agent
from pathlib import Path
from chunker import Chunker
from fastapi import FastAPI
from ai_client import AIClient
from db_client import DBClient
from dotenv import load_dotenv
from open_ai_routes import OpenAIRoutes
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool


def build_agents() -> list[Agent]:
    db_client = DBClient(
        host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT"))
    )

    ai_client = AIClient(api_key=os.getenv("GEMINI_API_KEY"))

    chunker = Chunker(db_client, ai_client)

    agents_json = json.loads(
        (Path(os.getenv("DEFINE_DIR")) / "agents.json").read_text(encoding="utf-8")
    )

    agents = [
        Agent(
            name=(cleaned_name := agent_json["name"].strip().replace(" ", "_").lower()),
            instruction=agent_json["instruction"],
            ai_client=ai_client,
            chunker=chunker,
            knowledge_dir=f"{os.getenv('DEFINE_DIR')}/{cleaned_name}",
        )
        for agent_json in agents_json
    ]

    for agent in agents:
        agent.load()

    return agents


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agents = await run_in_threadpool(build_agents)
    yield


load_dotenv()
app = FastAPI(title="HomeHeyAI", lifespan=lifespan)
app.include_router(OpenAIRoutes().router)
uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FASTAPI_PORT")))
