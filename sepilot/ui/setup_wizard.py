"""First-run setup wizard for SE Pilot interactive mode.

Guides the user through initial model configuration:
1. Enter OpenAI-compatible API base URL
2. Fetch available models from /v1/models
3. Select a model with arrow keys
4. Optionally enter an API key
5. Save as default profile
"""

import os
from typing import Any

import httpx
from rich.console import Console

try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


def _fetch_models(base_url: str, api_key: str | None = None) -> list[dict[str, Any]]:
    """Fetch model list from OpenAI-compatible /v1/models endpoint."""
    import re

    url = base_url.rstrip("/")
    # If URL already ends with /models, use as-is
    if url.endswith("/models"):
        pass
    # If URL already ends with a versioned path (e.g. /v1, /v4, /api/paas/v4), just append /models
    elif re.search(r"/v\d+$", url):
        url += "/models"
    else:
        url += "/v1/models"

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    models = data.get("data", [])

    # Sort by id
    models.sort(key=lambda m: m.get("id", ""))
    return models


def _select_model_arrow_keys(
    models: list[dict[str, Any]], console: Console
) -> str | None:
    """Arrow-key model selector using prompt_toolkit."""
    if not models:
        return None

    model_ids = [m.get("id", "unknown") for m in models]
    selected_idx = [0]
    page_size = min(15, os.get_terminal_size().lines - 6)

    if not HAS_PROMPT_TOOLKIT:
        # Fallback: numbered list
        console.print("[bold cyan]Available models:[/bold cyan]")
        for i, mid in enumerate(model_ids, 1):
            console.print(f"  [cyan]{i:3d}.[/cyan] {mid}")
        try:
            choice = input("\nModel number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(model_ids):
                return model_ids[idx]
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        return None

    kb = KeyBindings()

    @kb.add(Keys.Up)
    @kb.add("k")
    def _up(event):
        selected_idx[0] = max(0, selected_idx[0] - 1)
        event.app.exit(result="__refresh__")

    @kb.add(Keys.Down)
    @kb.add("j")
    def _down(event):
        selected_idx[0] = min(len(model_ids) - 1, selected_idx[0] + 1)
        event.app.exit(result="__refresh__")

    @kb.add(Keys.PageUp)
    def _page_up(event):
        selected_idx[0] = max(0, selected_idx[0] - page_size)
        event.app.exit(result="__refresh__")

    @kb.add(Keys.PageDown)
    def _page_down(event):
        selected_idx[0] = min(len(model_ids) - 1, selected_idx[0] + page_size)
        event.app.exit(result="__refresh__")

    @kb.add(Keys.Home)
    def _home(event):
        selected_idx[0] = 0
        event.app.exit(result="__refresh__")

    @kb.add(Keys.End)
    def _end(event):
        selected_idx[0] = len(model_ids) - 1
        event.app.exit(result="__refresh__")

    @kb.add(Keys.Enter)
    def _enter(event):
        event.app.exit(result="__select__")

    @kb.add(Keys.Escape)
    @kb.add(Keys.ControlC)
    def _cancel(event):
        event.app.exit(result="__cancel__")

    # Type-to-search buffer
    search_buf = [""]

    @kb.add("/")
    def _start_search(event):
        search_buf[0] = ""
        event.app.exit(result="__search__")

    while True:
        # Calculate visible window
        total = len(model_ids)
        half_page = page_size // 2
        start = max(0, selected_idx[0] - half_page)
        end = min(total, start + page_size)
        if end == total:
            start = max(0, total - page_size)

        # Render list
        lines = []
        for i in range(start, end):
            marker = ">" if i == selected_idx[0] else " "
            if i == selected_idx[0]:
                lines.append(f"  [bold cyan]{marker} {model_ids[i]}[/bold cyan]")
            else:
                lines.append(f"  [dim]{marker} {model_ids[i]}[/dim]")

        # Clear and redraw
        console.print()
        console.print(
            f"[bold]Select a model[/bold] [dim]({selected_idx[0]+1}/{total})[/dim]"
        )
        console.print("[dim]Arrow keys to navigate, Enter to select, / to search, Esc to cancel[/dim]")
        console.print()
        for line in lines:
            console.print(line)

        try:
            result = pt_prompt(
                HTML("<b>> </b>"),
                key_bindings=kb,
                default="",
            )
        except (EOFError, KeyboardInterrupt):
            return None

        if result == "__refresh__":
            # Move cursor up to redraw
            total_lines = len(lines) + 4  # header lines + model lines
            print(f"\033[{total_lines}A\033[J", end="", flush=True)
            continue
        elif result == "__select__":
            return model_ids[selected_idx[0]]
        elif result == "__cancel__":
            return None
        elif result == "__search__":
            # Inline search mode
            total_lines = len(lines) + 4
            print(f"\033[{total_lines}A\033[J", end="", flush=True)
            console.print()
            try:
                search_term = pt_prompt(
                    HTML("<b>Search: </b>"),
                )
            except (EOFError, KeyboardInterrupt):
                continue
            if search_term:
                search_lower = search_term.lower()
                for i, mid in enumerate(model_ids):
                    if search_lower in mid.lower():
                        selected_idx[0] = i
                        break
            # Redraw after search
            continue
        else:
            # Direct text input - treat as search
            if result and result.strip():
                search_lower = result.strip().lower()
                for i, mid in enumerate(model_ids):
                    if search_lower in mid.lower():
                        selected_idx[0] = i
                        break
            total_lines = len(lines) + 4
            print(f"\033[{total_lines}A\033[J", end="", flush=True)
            continue


def needs_setup() -> bool:
    """Check if first-run setup is needed.

    Returns True if no default profile exists AND no model-related
    environment variables are configured.
    """
    from sepilot.config.model_profile import ModelProfileManager

    # Check for default profile
    mgr = ModelProfileManager()
    if mgr.get_default_profile():
        return False

    # Check for common environment variables that indicate pre-configuration
    env_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_BASE",
        "LLM_BASE_URL",
        "API_BASE_URL",
        "SEPILOT_MODEL",
        "DEFAULT_MODEL",
    ]
    for key in env_keys:
        if os.getenv(key):
            return False

    return True


def run_setup_wizard(console: Console) -> dict[str, str | None] | None:
    """Run the first-run setup wizard.

    Returns:
        Dict with keys: base_url, model, api_key. Or None if cancelled.
    """
    from sepilot.config.model_profile import ModelProfileManager

    console.print()
    console.print("[bold cyan]Welcome to SE Pilot![/bold cyan]")
    console.print("[dim]LLM이 설정되지 않았습니다. 초기 설정을 시작합니다.[/dim]")
    console.print("[dim]Esc/Ctrl+C로 건너뛸 수 있습니다. (/model 로 나중에 설정 가능)[/dim]")
    console.print()

    # Step 1: API Base URL
    console.print("[bold]1. API Base URL[/bold]")
    console.print("[dim]   OpenAI 호환 API의 base URL을 입력하세요[/dim]")
    console.print("[dim]   예: http://localhost:11434, https://api.openai.com[/dim]")
    console.print()

    try:
        if HAS_PROMPT_TOOLKIT:
            base_url = pt_prompt(
                HTML("<b>   Base URL: </b>"),
                default="http://localhost:11434",
            )
        else:
            base_url = input("   Base URL [http://localhost:11434]: ").strip()
            if not base_url:
                base_url = "http://localhost:11434"
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]설정을 건너뜁니다.[/dim]")
        return None

    base_url = base_url.strip()
    if not base_url:
        base_url = "http://localhost:11434"

    # Step 2: API Key (optional, ask before fetching models)
    console.print()
    console.print("[bold]2. API Key[/bold] [dim](선택사항, 인증이 필요 없으면 Enter)[/dim]")

    try:
        if HAS_PROMPT_TOOLKIT:
            api_key = pt_prompt(
                HTML("<b>   API Key: </b>"),
                default="",
                is_password=True,
            )
        else:
            api_key = input("   API Key (Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]설정을 건너뜁니다.[/dim]")
        return None

    api_key = api_key.strip() or None

    # Step 3: Fetch and select model
    console.print()
    console.print("[bold]3. 모델 선택[/bold]")
    console.print(f"[dim]   {base_url} 에서 모델 목록을 가져오는 중...[/dim]")

    try:
        models = _fetch_models(base_url, api_key)
    except httpx.ConnectError:
        console.print(f"[red]   연결 실패: {base_url}[/red]")
        console.print("[dim]   URL을 확인하고 서버가 실행 중인지 확인하세요.[/dim]")
        return None
    except httpx.HTTPStatusError as e:
        console.print(f"[red]   HTTP 오류: {e.response.status_code}[/red]")
        if e.response.status_code == 401:
            console.print("[dim]   API Key가 필요합니다.[/dim]")
        return None
    except Exception as e:
        console.print(f"[red]   모델 목록 조회 실패: {e}[/red]")
        return None

    if not models:
        console.print("[yellow]   사용 가능한 모델이 없습니다.[/yellow]")
        return None

    console.print(f"[green]   {len(models)}개 모델 발견[/green]")

    selected_model = _select_model_arrow_keys(models, console)
    if not selected_model:
        console.print("\n[dim]모델 선택이 취소되었습니다.[/dim]")
        return None

    console.print(f"\n[green]   선택된 모델: {selected_model}[/green]")

    # Step 4: Save as default profile
    console.print()
    result = {
        "base_url": base_url,
        "model": selected_model,
        "api_key": api_key,
    }

    try:
        mgr = ModelProfileManager()
        mgr.set_parameter("base_url", base_url)
        mgr.set_parameter("model", selected_model)
        mgr.set_parameter("max_tokens", "16000")
        mgr.set_parameter("temperature", "0.3")
        if api_key:
            mgr.set_parameter("api_key", api_key)
        mgr.save_profile("default-setup")
        mgr.set_default_profile("default-setup")
        console.print("[green]설정이 'default-setup' 프로필로 저장되었습니다.[/green]")
    except Exception as e:
        console.print(f"[yellow]프로필 저장 실패: {e}[/yellow]")

    console.print()
    return result
