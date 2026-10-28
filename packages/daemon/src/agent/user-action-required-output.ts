export function userActionRequiredOutput(code: unknown, diagnostic: string, input: string, detail?: unknown): string {
  if (code === 'browser_control_required' && typeof detail === 'string' && detail.trim()) {
    return `${detail.trim().slice(0, 1000)}\n\n${userActionRequiredOutput(code, diagnostic, input)}`
  }
  const korean = /[가-힣]/u.test(input)
  if (code === 'workspace_execution_unavailable') {
    return korean
      ? [
          '폴더 격리 상태에서 명령을 실행할 격리 실행기가 없어 요청한 작업을 실행하지 못했습니다. 차단된 명령은 시작되지 않았습니다.',
          '현재 환경에서는 승인만으로 이 제한을 해제할 수 없습니다. 격리 실행기를 구성하거나, 직접 허용할 경우 이 대화의 작업공간 실행 범위를 로컬 실행 허용으로 변경한 뒤 다시 요청해 주세요. 로컬 실행은 폴더 밖 접근을 OS 수준에서 격리하지 않습니다.',
          '현재 실행 범위는 변경하지 않았으며, 다른 도구나 에이전트로 우회 재시도하지 않습니다.',
          `차단 사유: ${diagnostic}`,
        ].join('\n\n')
      : [
          'The requested operation could not run: folder isolation is enabled but no confined command or background-process runner is available. The blocked command was not started.',
          'Approval alone cannot remove this boundary. Configure a confined runner, or explicitly allow local execution for this conversation and retry. Local execution does not provide OS-level isolation from files outside the workspace.',
          'The execution boundary was not changed. No retry through another tool or agent will be attempted.',
          `Policy reason: ${diagnostic}`,
        ].join('\n\n')
  }
  if (code === 'browser_connection_required') {
    return korean
      ? [
          '이 대화에 브라우저 탭이 아직 연결되지 않았어요. 공유한 탭에서 작업하려면 대화 연결과 제어 허용이 필요합니다.',
          '1. **설정 → Connections → 브라우저 공동 탐색**을 여세요. 이미 확장에서 “이 탭 연결”을 눌렀다면 탭 공유만 완료된 상태입니다. 공유된 탭 목록에서 **이 대화에 연결**을 누른 뒤 **에이전트 제어 허용**을 눌러 주세요.',
          '2. 아직 공유한 탭이 없다면 **Chrome 연결 (확장 없음)**을 눌러 탭을 고르세요. Chrome 144 이상에서 `chrome://inspect/#remote-debugging`의 원격 디버깅을 켜고, Chrome이 허용을 물으면 **허용**을 누르세요. 탭 선택 후 **에이전트 제어 허용**도 필요합니다.',
          '다른 방법: Edge나 Chrome 143 이하는 **다른 방법: 브라우저 확장 사용**에서 확장과 연결 토큰으로 탭을 공유한 뒤, 같은 패널에서 **이 대화에 연결 → 에이전트 제어 허용**을 완료하세요.',
          '연결이 끝나면 이 대화에 “이어서 진행해 줘”라고 말씀해 주세요. 연결된 탭을 확인하고 이어서 진행할게요.',
        ].join('\n\n')
      : [
          'No browser tab is attached to this chat yet. Working in a shared tab requires attaching it to this chat and enabling agent control.',
          '1. Open **Settings → Connections → 브라우저 공동 탐색**. If you already clicked “이 탭 연결” in the extension, tab sharing is complete but chat attachment is still required. Click **이 대화에 연결** beside the shared tab, then **에이전트 제어 허용**.',
          '2. If no tab is shared yet, click **Chrome 연결 (확장 없음)** and pick a tab. Chrome 144+ must have remote debugging enabled at `chrome://inspect/#remote-debugging`; click **Allow** when Chrome asks. Then click **에이전트 제어 허용**.',
          '다른 방법 (alternative): on Edge or Chrome 143 and older, open **다른 방법: 브라우저 확장 사용** to share a tab with the extension and a connection token, then complete **이 대화에 연결 → 에이전트 제어 허용** in the same panel.',
          'Then tell me to continue in this chat. I will inspect the connected tab before proceeding.',
        ].join('\n\n')
  }
  if (code === 'browser_control_required') {
    return korean
      ? '브라우저는 이 대화에 연결됐지만 아직 직접 조작 모드라 제가 화면을 확인하거나 조작할 수 없어요.\n\n직접 조작을 마치면 **설정 → Connections → 브라우저 공동 탐색**에서 **에이전트 제어 허용**을 누르고, 이 대화에 이어서 진행하라고 말씀해 주세요.'
      : 'The browser is attached to this chat, but it is still under your control, so I cannot inspect or interact with the page.\n\nWhen you are ready, open **Settings → Connections → 브라우저 공동 탐색**, click **에이전트 제어 허용**, and tell me to continue in this chat.'
  }
  return diagnostic
}
