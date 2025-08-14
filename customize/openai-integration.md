# gemini-cli OpenAI 통합 가이드

## 설정 방법

1. OpenAI API 키 설정:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. OpenAI 호환 API용 커스텀 베이스 URL 설정 (선택사항):
```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 선택사항, 기본값은 OpenAI
```

3. 모델 및 임베딩 모델 설정 (선택사항):
```bash
export OPENAI_MODEL="gpt-4o"  # 기본값 (중요: OpenAI 모델명을 사용해야 함)
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"  # 기본값
```

⚠️ **중요**: `OPENAI_MODEL`에는 반드시 OpenAI 호환 모델명을 사용해야 합니다:
- ✅ 올바른 예: `gpt-4o`, `gpt-4-turbo-preview`, `gpt-3.5-turbo`
- ❌ 잘못된 예: `gemini-2.5-pro`, `claude-3-opus`

## OpenAI로 gemini-cli 실행하기

CLI 시작:
```bash
gemini
```

1. 인증 메시지가 나타나면 "Use OpenAI API Key"를 선택하세요.
2. 그 다음 사용 가능한 모델 목록이 표시됩니다. 원하는 모델을 선택하세요.
   - `/v1/models` API를 통해 자동으로 사용 가능한 모델을 가져옵니다
   - `OPENAI_MODEL` 환경 변수가 설정되어 있으면 기본값으로 사용됩니다

## 대안: 직접 프롬프트 모드

```bash
gemini -p "안녕하세요, 코딩을 도와주실 수 있나요?"
```

## 지원되는 기능

- ✅ 채팅 완성
- ✅ 스트리밍 응답
- ✅ 함수 호출 (도구)
- ✅ 토큰 카운팅
- ✅ 임베딩
- ✅ 커스텀 베이스 URL (LocalAI, Ollama 등 OpenAI 호환 API 지원)
- ✅ 자동 모델 탐색 (/v1/models API를 통한 사용 가능한 모델 목록 가져오기)
- ✅ 모델 선택 다이얼로그

## 설정 파일을 통한 구성

설정 파일에서도 OpenAI를 구성할 수 있습니다:

```json
{
  "useOpenAI": true,
  "openAIApiKey": "your-api-key",
  "openAIModel": "gpt-4o",
  "openAIBaseURL": "https://api.openai.com/v1",
  "openAIEmbeddingModel": "text-embedding-3-small"
}
```

## 제공자 간 전환

CLI는 마지막 인증 선택을 기억합니다. Google Gemini와 OpenAI 간 전환 방법:
1. CLI에서 `/auth` 명령 사용
2. 설정 파일 편집
3. 설정에서 인증 선택을 삭제하여 다시 프롬프트 받기

## 지원되는 OpenAI 호환 서비스

이 통합은 OpenAI API와 호환되는 다양한 서비스를 지원합니다:
- OpenAI (GPT-4, GPT-3.5 등)
- Azure OpenAI Service
- LocalAI
- Ollama (OpenAI 호환 모드)
- LM Studio
- 기타 OpenAI API 호환 서비스

## 사용 예시

### 기본 OpenAI 사용
```bash
export OPENAI_API_KEY="sk-..."
gemini
# "Use OpenAI API Key" 선택
```

### LocalAI 사용
```bash
export OPENAI_API_KEY="dummy-key"  # LocalAI는 보통 API 키가 필요 없음
export OPENAI_BASE_URL="http://localhost:8080/v1"
export OPENAI_MODEL="local-model-name"
gemini
```

### Azure OpenAI 사용
```bash
export OPENAI_API_KEY="your-azure-api-key"
export OPENAI_BASE_URL="https://your-resource.openai.azure.com"
export OPENAI_MODEL="your-deployment-name"
gemini
```

## 문제 해결

### API 키가 인식되지 않는 경우
- 환경 변수가 올바르게 설정되었는지 확인
- `.env` 파일을 사용하는 경우 올바른 위치에 있는지 확인

### 연결 오류
- 네트워크 연결 확인
- 프록시 설정이 필요한 경우 `HTTPS_PROXY` 환경 변수 설정
- 커스텀 베이스 URL이 올바른지 확인

### 모델을 찾을 수 없는 경우
- 사용하려는 모델이 API 키로 접근 가능한지 확인
- 모델 이름이 정확한지 확인 (예: `gpt-4o`, `gpt-3.5-turbo`)

### JSON 응답 파싱 오류
OpenAI API는 Gemini와 다른 방식으로 구조화된 JSON을 생성합니다. 이 통합은 OpenAI의 JSON 모드를 자동으로 활성화하여 이 문제를 해결합니다. 만약 여전히 오류가 발생한다면:

1. 프로젝트 재빌드:
```bash
cd /path/to/gemini-cli
npm run clean
npm run build
npm link
```

2. GPT-4 모델 사용 권장 (JSON 생성이 더 안정적임):
```bash
export OPENAI_MODEL="gpt-4-turbo-preview"
```

3. 디버그 모드로 실행하여 상세 정보 확인:
```bash
DEBUG=1 gemini -d
```

4. 캐시 초기화:
```bash
rm -rf ~/.gemini-cli/cache
rm -rf ~/.config/gemini-cli
```

참고: OpenAI API는 Gemini의 JSON 모드와 다르게 동작하므로, 일부 고급 기능(자동 대화 연속성 확인 등)에서 간헐적인 오류가 발생할 수 있습니다. 이는 정상적인 채팅 기능에는 영향을 주지 않습니다.