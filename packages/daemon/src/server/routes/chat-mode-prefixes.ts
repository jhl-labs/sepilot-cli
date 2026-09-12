import { getDocRegistry } from '../../agent/doc/session.js'
import { parseOutline } from '../../agent/doc/parse.js'
import { strictWorkspacePathViolation } from '../../security/policy-engine.js'

/**
 * Shell 모드(mode='shell') 시스템 프롬프트 prefix.
 * desktop의 Shell AI 패널은 LLM 응답 안의 ```bash / ```sh / ```pwsh 등 펜스
 * 코드블록을 자동 추출해 활성 터미널에 Run 버튼으로 dispatch 한다. LLM이
 * 그 포맷에 맞게 응답하지 않으면 추출 정확도가 떨어지고 자동 실행 토글이
 * 무의미해지므로, 모드 진입 시점에 응답 형식을 명시적으로 가이드한다.
 */
export function buildShellModePrefix(mode: string | undefined): string {
  if (mode !== 'shell') return ''
  return [
    '## Shell 모드',
    '사용자는 desktop의 Shell AI 패널에서 활성 터미널(local 또는 SSH)을 옆에 두고 명령을 묻고 있다. 다음 규칙을 지켜라:',
    '',
    '1. **명령은 항상 펜스 코드블록으로**. 셸 명령을 제안할 때는 반드시',
    '   ```bash / ```sh / ```zsh / ```pwsh / ```cmd 중 사용자 OS에 맞는 lang hint를',
    '   적어라. 산문 안에 백틱 한 개로 적힌 명령은 데스크톱이 못 잡아낸다.',
    '2. **한 블록 = 한 명령 (또는 한 묶음)**. 사용자가 Run 버튼을 누르면 그 블록',
    '   전체가 활성 터미널 stdin 으로 들어간다. 여러 step 이면 step 마다 별도',
    '   블록으로 끊어라. 한 블록 안에 줄바꿈으로 묶을 땐 정말 한 호흡에 같이',
    '   실행되어야 하는 것만.',
    '3. **prompt 마커 ($ , > , # ) 붙이지 마라**. 블록 안에 들어간 텍스트가 그대로',
    '   터미널로 전송되므로, `$ ls -la` 가 아니라 `ls -la` 만 적는다 (데스크톱이',
    '   안전망으로 prefix 를 떼긴 하지만 처음부터 안 붙는 게 정답).',
    '4. **산문은 짧게**. 명령 한두 줄 위에 "왜 이 명령" 한 줄, 아래에 "기대 출력 /',
    '   주의사항" 한두 줄이면 충분하다. 사용자가 빠르게 읽고 ▶ 를 누르는 흐름.',
    '5. **위험 명령은 명시적으로 경고**. rm -rf, git clean -fdx, docker system',
    '   prune -f, kubectl delete --all, terraform destroy, dd of=/dev/..., mkfs,',
    '   chmod -R 777 / 같은 패턴이라면 블록 앞 줄에 "⚠️" 표시와 한 줄 설명을',
    '   덧붙여라. 데스크톱이 별도 확인 다이얼로그를 띄우지만 사용자에게 미리',
    '   경고하는 책임은 너에게 있다.',
    '6. **계획·진단 질문엔 명령 없이도 답해도 된다**. "이 에러 무슨 뜻이야"',
    '   같은 질문엔 블록 없이 산문만 줘도 OK. 강제로 명령을 만들지 마라.',
    '',
    '---',
    '',
  ].join('\n')
}

/**
 * 글쓰기 모드(mode='writing') 자동 컨텍스트 prefix.
 * 활성 doc이 있을 때 system prompt에 outline + (작은 doc) 본문 + 도구 안내를
 * prepend. 사용자가 별도 button 안 눌러도 LLM이 활성 문서를 인지하고
 * doc.outline/doc.replace_section 등을 호출하게 함.
 */
export function buildWritingDocPrefix(
  mode: string | undefined,
  writingDocId: string | undefined,
  workspaceRoot?: string,
): string {
  if (mode !== 'writing') return ''
  const registry = getDocRegistry()
  const id = writingDocId ?? registry.getActiveId()
  if (!id) return ''
  const session = registry.get(id)
  if (!session) return ''
  if (
    session.path
    && workspaceRoot
    && strictWorkspacePathViolation(session.path, workspaceRoot)
  ) {
    return [
      '## 활성 글쓰기 문서 사용 불가',
      '선택한 문서가 현재 워크스페이스 밖에 있어 본문을 컨텍스트에 첨부하지 않았습니다.',
      '이 문서를 읽거나 수정하지 말고, 사용자에게 문서가 포함된 워크스페이스를 선택하거나 워크스페이스 안의 문서를 열어 달라고 안내하세요.',
      '',
      '---',
      '',
    ].join('\n')
  }
  const outline = parseOutline(session.content)
  const outlineText = outline.length
    ? outline
        .map((e) => `[${e.index}] ${'#'.repeat(e.level)} ${e.title}  (chars ${e.start}..${e.end})`)
        .join('\n')
    : '(no headings)'
  // 큰 문서는 outline + 앞 4000자만, 작은 문서는 전체.
  const INLINE_LIMIT = 8000
  const inlineBody =
    session.content.length <= INLINE_LIMIT
      ? session.content
      : session.content.slice(0, 4000) +
        `\n\n[...문서가 길어 앞 4000자만 표시. 필요시 doc.get(section_index)로 특정 섹션을 가져오세요...]`
  const pathInfo = session.path ? `path=${session.path}` : '(저장 안 된 새 문서)'
  return [
    '## 활성 글쓰기 문서 (writing canvas)',
    `사용자는 desktop의 글쓰기 모드를 사용 중이며, 다음 문서를 함께 편집하고 있습니다. ${pathInfo}, version=${session.version}.`,
    '',
    '이 모드에서 사용자의 요청은 기본적으로 "채팅으로 설명해 달라"가 아니라 "오른쪽 활성 문서에 작성/수정해 달라"는 뜻입니다. 특히 "글을 써줘", "정리해줘", "초안 작성", "이어 써줘", "문서에 추가" 같은 요청은 반드시 활성 문서 본문을 변경하세요. 문서가 비어 있으면 doc.rewrite로 완성된 초안을 넣고, 기존 본문이 있으면 요청 의도에 맞게 doc.append/doc.replace_section/doc.replace_range를 사용하세요. 최종 채팅 답변은 변경 요약 한두 문장만 남기고, 작성한 본문을 채팅에 길게 복사하지 마세요.',
    '',
    '문서 편집은 반드시 doc.* 도구만 사용하세요. **활성 문서의 전체 본문과 outline이 이미 아래 컨텍스트에 첨부되어 있으므로** fs.glob / fs.read / fs.search / fs.edit / apply_patch / web.* 같은 도구로 디스크나 외부에서 문서를 찾으려 하지 마세요 — 절대 호출 금지. 그 도구들로 만든 변경은 desktop editor에 반영되지 않을 뿐 아니라, 사용자의 in-memory doc과 디스크가 갈라져 데이터 손실로 이어집니다.',
    '- 본문/섹션 읽기: doc.get(section?) — outline 인덱스 또는 heading 텍스트',
    '- outline 보기: doc.outline (이미 아래 첨부됨; 변경 후 재확인용으로 사용)',
    '- 부분 수정: doc.replace_section (heading 단위) 또는 doc.replace_range (정확 offset)',
    '- 추가: doc.append (문서 끝) / doc.insert_after_section (특정 섹션 뒤)',
    '- 전체 교체: doc.rewrite (큰 퇴고; 가능하면 doc.diff_preview를 먼저)',
    '- 큰 변경은 doc.diff_preview로 사용자 수락을 먼저 받기',
    '',
    '### 현재 outline',
    '```',
    outlineText,
    '```',
    '',
    '### 현재 본문',
    '```markdown',
    inlineBody,
    '```',
    '',
    '---',
    '',
  ].join('\n')
}
