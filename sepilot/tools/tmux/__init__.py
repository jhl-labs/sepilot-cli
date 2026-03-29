"""tmux 기반 에이전트 세션 관리.

sepilot이 tmux 세션을 통해 외부 CLI 에이전트(claude, opencode 등)를
대화형으로 제어하고 오케스트레이션하는 기능을 제공합니다.

Usage:
    from sepilot.tools.tmux import TmuxSessionManager

    manager = TmuxSessionManager()
    session_id = manager.create_session("claude", cwd="/path/to/project")
    manager.send_keys(session_id, "Fix the bug in auth.py")
    output = manager.wait_for_idle(session_id)
    manager.destroy_session(session_id)
"""

from sepilot.tools.tmux.tmux_agent_configs import TMUX_AGENT_CONFIGS
from sepilot.tools.tmux.tmux_session_manager import TmuxSession, TmuxSessionManager

__all__ = [
    "TMUX_AGENT_CONFIGS",
    "TmuxSession",
    "TmuxSessionManager",
]
