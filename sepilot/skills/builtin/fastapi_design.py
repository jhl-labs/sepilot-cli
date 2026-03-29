"""FastAPI Design Skill"""

from ..base import PromptSkill
from . import prompts


class FastAPIDesignSkill(PromptSkill):
    name = "fastapi-design"
    description = "Set up and develop FastAPI applications with uv package manager"
    triggers = [
        "fastapi", "fast api", "python api", "백엔드", "backend",
        "api 서버", "api server", "rest api", "restful",
        "python 서버", "python server", "uv", "uvicorn",
        "파이썬 백엔드", "파이썬 서버", "api 개발",
    ]
    category = "backend"
    prompt = prompts.load("fastapi_design")
