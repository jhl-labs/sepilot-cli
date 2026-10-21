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
          '이 대화에 브라우저가 아직 연결되지 않았어요. 페이지를 열거나 검색하려면 먼저 Chrome 탭을 연결해 주세요.',
          '1. 대화 상단의 **브라우저 공동 탐색**을 펼치고 **Chrome 연결 (확장 없음)**을 누르세요. Chrome 144 이상에서 `chrome://inspect/#remote-debugging`의 원격 디버깅이 켜져 있어야 하며, Chrome이 허용을 물으면 **허용**을 누르세요.',
          '2. 공유할 탭을 목록에서 고른 뒤 **에이전트 제어 허용**을 눌러 주세요.',
          '다른 방법: Edge나 Chrome 143 이하는 **다른 방법: 브라우저 확장 사용**에서 Browser Companion 확장과 연결 토큰으로 탭을 연결할 수 있어요.',
          '연결이 끝나면 이 대화에 “이어서 진행해 줘”라고 말씀해 주세요. 연결된 탭을 확인하고 이어서 진행할게요.',
        ].join('\n\n')
      : [
          'No browser is connected to this chat yet. Connect a Chrome tab before I can open a page or search.',
          '1. Expand **브라우저 공동 탐색** at the top of the chat and click **Chrome 연결 (확장 없음)**. Chrome 144+ must have remote debugging enabled at `chrome://inspect/#remote-debugging`; click **Allow** when Chrome asks.',
          '2. Pick the tab to share from the list, then click **에이전트 제어 허용**.',
          '다른 방법 (alternative): on Edge or Chrome 143 and older, open **다른 방법: 브라우저 확장 사용** and connect the tab through the Browser Companion extension with a connection token.',
          'Then tell me to continue in this chat. I will inspect the connected tab before proceeding.',
        ].join('\n\n')
  }
  if (code === 'browser_control_required') {
    return korean
      ? '지금은 브라우저를 직접 조작 중이라 제가 화면을 확인하거나 조작할 수 없어요.\n\n직접 조작을 마치면 대화 상단의 **브라우저 공동 탐색**에서 **에이전트 제어 허용**을 누르고, 이 대화에 “이어서 진행해 줘”라고 말씀해 주세요. 제어권을 돌려주실 때까지 기다릴게요.'
      : 'You currently have control of the browser, so I cannot inspect or interact with the page.\n\nWhen you are ready, open **브라우저 공동 탐색**, click **에이전트 제어 허용**, and tell me to continue in this chat. I will wait until you return control.'
  }
  return diagnostic
}
