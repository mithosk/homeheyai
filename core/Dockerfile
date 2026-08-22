FROM python:3.14-slim

WORKDIR /project
COPY . .

RUN pip install uv
RUN uv sync --frozen

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "src/main.py"]