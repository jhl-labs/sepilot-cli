"""전문화된 SubAgent 구현

각 도메인별로 특화된 SubAgent들을 제공합니다.
"""

import logging

from langchain_core.language_models import BaseChatModel

from .base_subagent import BaseSubAgent
from .models import SubAgentTask

logger = logging.getLogger(__name__)


class AnalyzerSubAgent(BaseSubAgent):
    """코드 분석 전문 SubAgent

    코드 구조 분석, 복잡도 계산, import 분석 등을 전문으로 처리합니다.

    사용 가능한 도구:
        - code_analyze: AST 기반 코드 분석
        - file_read: 파일 읽기
        - codebase: 코드베이스 검색
    """

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="analyzer",
            tools=tools or [],
            llm=llm
        )

        # 분석 관련 키워드
        self.keywords = [
            "분석", "analyze", "복잡도", "complexity",
            "구조", "structure", "함수", "function",
            "클래스", "class", "import", "의존성", "dependency"
        ]

    def can_handle(self, task: SubAgentTask) -> bool:
        """분석 관련 작업 처리 가능 여부 판단"""
        task_desc_lower = task.description.lower()

        # 키워드 매칭
        keyword_match = any(
            keyword.lower() in task_desc_lower
            for keyword in self.keywords
        )

        # agent_type 지정 확인
        type_match = task.agent_type == "analyzer"

        return keyword_match or type_match

    async def _execute_task(self, task: SubAgentTask) -> str:
        """분석 작업 실행"""
        logger.info(f"[AnalyzerSubAgent] Executing analysis task: {task.task_id}")

        # code_analyze 도구 확인
        code_analyze_tool = self.get_tool("code_analyze")
        _file_read_tool = self.get_tool("file_read")  # noqa: F841

        if code_analyze_tool and "file_path" in task.context:
            # code_analyze 도구 사용
            file_path = task.context["file_path"]

            try:
                result = code_analyze_tool.invoke({
                    "action": "analyze_file",
                    "file_path": file_path
                })
                return f"✅ 파일 분석 완료:\n{result}"
            except Exception as e:
                logger.error(f"code_analyze failed: {e}")

        # LLM 기반 분석
        if self.llm:
            system_prompt = """You are a specialized code analysis agent.

# Task Type: READ-ONLY (분석/설명)

# 필수 지침
- This is a READ-ONLY analysis task
- Use file_read tools to examine code
- DO NOT modify any files
- Provide detailed analysis in your response

# Your Expertise
- Code structure analysis (functions, classes, modules)
- Complexity measurement (cyclomatic complexity, nesting depth)
- Dependency analysis
- Code quality assessment

# Workflow
1. Read and examine the target code/files
2. Analyze structure, complexity, and quality
3. Respond with detailed, actionable analysis
4. Include specific recommendations (but do not implement them)

# 산출물
- Detailed analysis report (text response, not file modifications)"""

            response = await self._call_llm(task.description, system_prompt)
            return response

        return "❌ 분석 도구가 없습니다."


class CodeGenSubAgent(BaseSubAgent):
    """코드 생성 전문 SubAgent

    코드 작성, 파일 수정, 리팩토링 등을 전문으로 처리합니다.

    사용 가능한 도구:
        - file_write: 파일 쓰기
        - file_edit: 파일 편집
        - code_analyze: 코드 검증용
    """

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="codegen",
            tools=tools or [],
            llm=llm
        )

        self.keywords = [
            "생성", "create", "작성", "write",
            "코드", "code", "함수", "function",
            "클래스", "class", "수정", "modify",
            "리팩토링", "refactor"
        ]

    def can_handle(self, task: SubAgentTask) -> bool:
        """코드 생성 관련 작업 처리 가능 여부 판단"""
        task_desc_lower = task.description.lower()

        keyword_match = any(
            keyword.lower() in task_desc_lower
            for keyword in self.keywords
        )

        type_match = task.agent_type == "codegen"

        return keyword_match or type_match

    async def _execute_task(self, task: SubAgentTask) -> str:
        """코드 생성 작업 실행"""
        logger.info(f"[CodeGenSubAgent] Executing code generation task: {task.task_id}")

        # LLM을 사용하여 코드 생성
        if self.llm:
            system_prompt = """You are a specialized code generation agent.

# Task Type: MODIFICATION (작업/생성)

# 필수 지침
- This is a code GENERATION task - you MUST create actual files
- Use file_write or file_edit tools to create/modify code files
- Do NOT just describe the code - write it to files
- Verify the files were created successfully

# Your Expertise
- Writing clean, efficient code
- Following best practices and coding standards
- Creating well-structured functions and classes
- Refactoring existing code

# Workflow
1. Understand the code generation requirements
2. Use file_write to create new code files
3. Or use file_edit to modify existing files
4. Verify the code was written successfully
5. Respond with summary of what was created

# 산출물
- Actual code files (created via file_write/file_edit)
- Summary of generated code

Always generate production-ready code with proper documentation."""

            response = await self._call_llm(task.description, system_prompt)
            return response

        return "❌ LLM이 설정되지 않았습니다."


class TestingSubAgent(BaseSubAgent):
    """테스트 전문 SubAgent

    테스트 코드 생성, 테스트 실행, 결과 검증 등을 전문으로 처리합니다.

    사용 가능한 도구:
        - bash_execute: 테스트 실행
        - file_write: 테스트 파일 작성
        - file_read: 테스트 대상 파일 읽기
    """

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="testing",
            tools=tools or [],
            llm=llm
        )

        self.keywords = [
            "테스트", "test", "실행", "run",
            "검증", "verify", "assert", "pytest",
            "unittest", "coverage"
        ]

    def can_handle(self, task: SubAgentTask) -> bool:
        """테스트 관련 작업 처리 가능 여부 판단"""
        task_desc_lower = task.description.lower()

        keyword_match = any(
            keyword.lower() in task_desc_lower
            for keyword in self.keywords
        )

        type_match = task.agent_type == "testing"

        return keyword_match or type_match

    async def _execute_task(self, task: SubAgentTask) -> str:
        """테스트 작업 실행"""
        logger.info(f"[TestingSubAgent] Executing testing task: {task.task_id}")

        # bash_execute 도구로 테스트 실행
        bash_tool = self.get_tool("bash_execute")

        if bash_tool and "test_command" in task.context:
            test_command = task.context["test_command"]

            try:
                result = bash_tool.invoke({"command": test_command})
                return f"✅ 테스트 실행 완료:\n{result}"
            except Exception as e:
                logger.error(f"Test execution failed: {e}")
                return f"❌ 테스트 실행 실패: {e}"

        # LLM을 사용하여 테스트 생성/분석
        if self.llm:
            system_prompt = """You are a specialized testing agent.

# Task Type: MODIFICATION (테스트 작성)

# 필수 지침
- This is a test CREATION task - you MUST create actual test files
- Use file_write or file_edit to create test files
- Do NOT just describe tests - write them to files
- Run the tests if possible to verify they work

# Your Expertise
- Writing comprehensive test cases
- Test-driven development (TDD)
- Unit testing, integration testing
- Test coverage analysis
- Test result interpretation

# Workflow
1. Understand the code that needs testing
2. Use file_write to create test files (e.g., test_*.py)
3. Write comprehensive test cases with assertions
4. If possible, use bash_execute to run tests
5. Respond with summary of tests created

# 산출물
- Actual test files (created via file_write)
- Test execution results (if run)

Generate thorough, maintainable tests."""

            response = await self._call_llm(task.description, system_prompt)
            return response

        return "❌ 테스트 도구가 없습니다."


class DocumentationSubAgent(BaseSubAgent):
    """문서화 전문 SubAgent

    문서 생성, README 작성, API 문서화 등을 전문으로 처리합니다.

    사용 가능한 도구:
        - file_write: 문서 파일 작성
        - code_analyze: 코드 분석 (문서화 대상)
        - file_read: 기존 문서 읽기
    """

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="documentation",
            tools=tools or [],
            llm=llm
        )

        self.keywords = [
            "문서", "documentation", "doc", "readme",
            "설명", "explain", "보고서", "report",
            "가이드", "guide", "api", "주석", "comment"
        ]

    def can_handle(self, task: SubAgentTask) -> bool:
        """문서화 관련 작업 처리 가능 여부 판단"""
        task_desc_lower = task.description.lower()

        keyword_match = any(
            keyword.lower() in task_desc_lower
            for keyword in self.keywords
        )

        type_match = task.agent_type == "documentation"

        return keyword_match or type_match

    async def _execute_task(self, task: SubAgentTask) -> str:
        """문서화 작업 실행"""
        logger.info(f"[DocumentationSubAgent] Executing documentation task: {task.task_id}")

        # LLM을 사용하여 문서 생성
        if self.llm:
            system_prompt = """You are a specialized documentation agent.

# Task Type: MODIFICATION (문서 작성)

# 필수 지침
- This is a documentation CREATION task - you MUST create/update actual files
- Use file_write to create documentation files (README.md, docs/*.md)
- Use file_edit to add docstrings to code files
- Do NOT just describe the documentation - write it to files
- Verify the documentation files were created

# Your Expertise
- Writing clear, comprehensive documentation
- Creating user guides and tutorials
- API documentation
- Code comments and docstrings
- README files

# Workflow
1. Read the code that needs documentation
2. Use file_write to create documentation files (e.g., README.md)
3. Or use file_edit to add docstrings to code files
4. Verify the documentation was created successfully
5. Respond with summary of documentation created

# 산출물
- Documentation files (README.md, docs/*.md)
- Or updated code files with docstrings

Generate well-structured, accessible documentation."""

            response = await self._call_llm(task.description, system_prompt)
            return response

        return "❌ LLM이 설정되지 않았습니다."


class RefactoringSubAgent(BaseSubAgent):
    """리팩토링 전문 SubAgent

    코드 개선, 최적화, 구조 변경 등을 전문으로 처리합니다.

    사용 가능한 도구:
        - code_analyze: 복잡도 분석
        - file_edit: 코드 수정
        - file_read: 원본 코드 읽기
    """

    def __init__(
        self,
        agent_id: str,
        tools: list | None = None,
        llm: BaseChatModel | None = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="refactoring",
            tools=tools or [],
            llm=llm
        )

        self.keywords = [
            "리팩토링", "refactor", "개선", "improve",
            "최적화", "optimize", "정리", "clean",
            "단순화", "simplify", "restructure"
        ]

    def can_handle(self, task: SubAgentTask) -> bool:
        """리팩토링 관련 작업 처리 가능 여부 판단"""
        task_desc_lower = task.description.lower()

        keyword_match = any(
            keyword.lower() in task_desc_lower
            for keyword in self.keywords
        )

        type_match = task.agent_type == "refactoring"

        return keyword_match or type_match

    async def _execute_task(self, task: SubAgentTask) -> str:
        """리팩토링 작업 실행"""
        logger.info(f"[RefactoringSubAgent] Executing refactoring task: {task.task_id}")

        # code_analyze로 현재 상태 분석
        code_analyze_tool = self.get_tool("code_analyze")

        if code_analyze_tool and "file_path" in task.context:
            file_path = task.context["file_path"]

            try:
                # 복잡도 체크
                complexity_result = code_analyze_tool.invoke({
                    "action": "check_complexity",
                    "file_path": file_path,
                    "threshold": 5
                })

                # LLM으로 리팩토링 제안 및 실행
                if self.llm:
                    system_prompt = """You are a specialized refactoring agent.

# Task Type: MODIFICATION (리팩토링)

# 필수 지침
- This is a code REFACTORING task - you MUST modify actual files
- Use file_read to understand current code
- Use file_edit to apply refactoring changes
- Do NOT just suggest refactorings - implement them
- Verify the refactored code works correctly

# Your Expertise
- Code quality improvement
- Complexity reduction
- Design pattern application
- Performance optimization
- Maintainability enhancement

# Workflow
1. Read the code file using file_read
2. Analyze complexity and quality issues
3. Use file_edit to apply refactoring changes
4. Verify the changes (read the file again or run tests)
5. Respond with summary of refactorings applied

# 산출물
- Refactored code files (modified via file_edit)
- Summary of improvements made

Provide specific, actionable refactorings and IMPLEMENT them."""

                    prompt = f"""다음 파일을 리팩토링해야 합니다:

파일: {file_path}

복잡도 분석 결과:
{complexity_result}

리팩토링을 적용하고 파일을 수정해주세요."""

                    response = await self._call_llm(prompt, system_prompt)
                    return response

                return f"복잡도 분석 완료:\n{complexity_result}"

            except Exception as e:
                logger.error(f"Refactoring analysis failed: {e}")

        # LLM만 사용
        if self.llm:
            system_prompt = """You are a specialized refactoring agent.

# Task Type: MODIFICATION (리팩토링)

# 필수 지침
- This is a code REFACTORING task - you MUST modify actual files
- Use file_read to examine code
- Use file_edit to apply refactorings
- Do NOT just suggest - IMPLEMENT the refactorings
- Verify changes work correctly

# Analyze and improve:
- Code complexity
- Readability
- Maintainability
- Performance
- Best practices

# Workflow
1. Read the target code files
2. Identify improvement opportunities
3. Apply refactorings using file_edit
4. Verify the refactored code
5. Respond with summary of changes made

# 산출물
- Refactored code files (via file_edit)
- Summary of improvements"""

            response = await self._call_llm(task.description, system_prompt)
            return response

        return "❌ 리팩토링 도구가 없습니다."
