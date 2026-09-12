/** Present attested prerequisites without exposing diagnostic protocol text.
 * Codes select the recovery steps; language affects presentation only.
 */
export function userActionRequiredOutput(code: unknown, diagnostic: string, input: string, detail?: unknown): string {
  if (code === 'browser_control_required' && typeof detail === 'string' && detail.trim()) {
    return `${detail.trim().slice(0, 1000)}\n\n${userActionRequiredOutput(code, diagnostic, input)}`
  }
  const korean = /[가-힣]/u.test(input)
  if (code === 'browser_connection_required') {
    return korean
      ? [
          '이 대화에 브라우저가 아직 연결되지 않았어요. 페이지를 열거나 검색하려면 먼저 Chrome 또는 Edge를 연결해 주세요.',
          '1. 대화 상단의 **브라우저 공동 탐색**을 펼쳐 주세요. 확장이 없다면 **Chrome에 설치** 또는 **Edge에 설치**를 눌러 안내를 따라 주세요.',
          '2. **연결 토큰 관리**에서 **연결 토큰 만들기**를 누르세요. 공유할 웹 탭에서 Browser Companion 확장을 열고 연결 주소(기본값: `http://127.0.0.1:17600`)와 토큰을 입력한 뒤 **이 탭 연결**을 눌러 주세요.',
          '3. Desktop으로 돌아와 표시된 브라우저를 이 대화에 연결하고 **에이전트 제어 허용**을 눌러 주세요.',
          '연결이 끝나면 이 대화에 “이어서 진행해 줘”라고 말씀해 주세요. 연결된 탭을 확인하고 이어서 진행할게요.',
        ].join('\n\n')
      : [
          'No browser is connected to this chat yet. Connect Chrome or Edge before I can open a page or search.',
          '1. Expand **브라우저 공동 탐색** at the top of the chat. If needed, use **Chrome에 설치** or **Edge에 설치** and follow the installation steps.',
          '2. Under **연결 토큰 관리**, click **연결 토큰 만들기**. Open Browser Companion in the web tab you want to share, enter the connection address (default: `http://127.0.0.1:17600`) and token, and click **이 탭 연결**.',
          '3. Return to Desktop, select that browser for this chat, and click **에이전트 제어 허용**.',
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
