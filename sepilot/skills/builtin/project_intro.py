"""Project Introduction Skill"""

from ..base import PromptSkill


class ProjectIntroSkill(PromptSkill):
    name = "project-intro"
    description = "Analyze project structure and generate introduction documentation"
    triggers = [
        "project intro", "project introduction", "introduce project",
        "project overview", "document project", "project documentation",
        "what is this project", "explain project",
    ]
    category = "documentation"
    prompt = """\
## Project Introduction Documentation Task

You are tasked with analyzing the current project and generating comprehensive introduction documentation.

### Step 1: Explore Project Structure
First, explore the project structure to understand its organization:
- Run `tree -L 3 -I 'node_modules|__pycache__|.git|dist|build|*.pyc|.mypy_cache|.pytest_cache|venv|.venv'` to see the directory structure
- Identify key directories and their purposes

### Step 2: Analyze Key Files
Read and analyze these important files (if they exist):
- `README.md` - Existing documentation
- `pyproject.toml` or `package.json` - Project metadata and dependencies
- `setup.py` or `setup.cfg` - Build configuration
- `Makefile` or `Dockerfile` - Build/deployment configuration
- Main entry points (`main.py`, `__main__.py`, `index.js`, `app.py`, etc.)
- Configuration files (`.env.example`, `config.py`, `settings.py`, etc.)

### Step 3: Identify Architecture
Look for patterns that indicate the architecture:
- Framework used (FastAPI, Flask, Django, React, Vue, etc.)
- Directory structure patterns (MVC, Clean Architecture, etc.)
- Key modules and their responsibilities

### Step 4: Generate Documentation
Create a comprehensive project introduction document that includes:

#### 1. Project Overview
- Project name and brief description
- Main purpose and use cases
- Target users or audience

#### 2. Technology Stack
- Programming languages used
- Main frameworks and libraries
- Database and storage solutions (if applicable)
- External services/APIs (if applicable)

#### 3. Project Structure
```
project/
├── directory1/    # Description
├── directory2/    # Description
└── ...
```
- Explain the purpose of each major directory
- Highlight key files and their roles

#### 4. Key Components
- List and describe main modules/packages
- Explain how components interact
- Include architecture diagram description if applicable

#### 5. Getting Started
- Prerequisites
- Installation steps
- Configuration requirements
- How to run the project

#### 6. Development
- Development setup
- Testing approach
- Code style/linting configuration
- Build process

### Output Format
Generate the documentation in Markdown format. The document should be:
- Clear and well-organized
- Suitable for new developers joining the project
- Technical but accessible
- Include code examples where helpful

**IMPORTANT**:
- Write the documentation to a file named `PROJECT_INTRO.md` in the project root
- Do NOT include sensitive information (API keys, passwords, internal IPs, usernames)
- Focus on publicly shareable information about the project structure and architecture"""
