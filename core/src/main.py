import os
from google import genai
from dotenv import load_dotenv
from chunker import refresh_chunks
from qdrant_client import QdrantClient
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.blocking import BlockingScheduler


def start_scheduler():
    scheduler = BlockingScheduler()

    qdrant_client = QdrantClient(
        host=os.getenv("QDRANT_HOST"),
        port=int(os.getenv("QDRANT_PORT") or 0),
        check_compatibility=False
    )

    gemini_client = genai.Client(
        api_key=os.getenv("LLM_API_KEY")
    )

    scheduler.add_job(
        refresh_chunks,
        CronTrigger.from_crontab(os.getenv("KNOWLEDGE_CRON")),
        args=[os.getenv("KNOWLEDGE_DIR"), qdrant_client, gemini_client],
    )

    scheduler.start()


load_dotenv()
start_scheduler()
