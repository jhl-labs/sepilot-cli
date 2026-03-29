"""Frontend Design Skill"""

from ..base import PromptSkill
from . import prompts


class FrontendDesignSkill(PromptSkill):
    name = "frontend-design"
    description = "Set up and develop modern frontend apps with Vite + Bun + TypeScript + React"
    triggers = [
        "frontend", "react", "vite", "web app", "웹 개발", "프론트엔드",
        "create react", "react app", "typescript react", "react project",
        "web service", "웹 서비스", "SPA", "single page", "UI 개발",
        "bun create", "vite project", "react setup", "web frontend",
    ]
    category = "frontend"
    prompt = prompts.load("frontend_design")
