import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage, printApiError } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const ANSWER_COPY = {
  en: {
    missingAnswer: 'Answer text is required.',
    answered: (questionId: string) => `Answered ${questionId}`,
    followUp: (sessionId: string) =>
      `  run: live; return to the waiting command, or inspect with: sepilot sessions show ${sessionId}`,
    failed: (questionId: string, msg: string) => `Failed to answer ${questionId}: ${msg}`,
  },
  ko: {
    missingAnswer: '답변 텍스트가 필요합니다.',
    answered: (questionId: string) => `${questionId} 답변 완료`,
    followUp: (sessionId: string) =>
      `  실행: live; 기다리던 명령으로 돌아가거나 확인하세요: sepilot sessions show ${sessionId}`,
    failed: (questionId: string, msg: string) => `${questionId} 답변 실패: ${msg}`,
  },
} as const

export interface AnswerCommandOptions {
  url?: string
}

export async function answerCommand(
  sessionId: string,
  questionId: string,
  answerParts: string[],
  options: AnswerCommandOptions = {},
): Promise<void> {
  const copy = ANSWER_COPY[detectCliLocale()] ?? ANSWER_COPY.en
  const answer = answerParts.join(' ').trim()
  if (!answer) {
    console.error(chalk.red(copy.missingAnswer))
    process.exit(1)
  }

  const client = new DaemonClient(options.url)
  try {
    const result = await client.answerSessionQuestion(sessionId, questionId, answer)
    output(result, () => [
      chalk.green(copy.answered(questionId)),
      chalk.gray(copy.followUp(sessionId)),
    ].join('\n'))
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(copy.failed(questionId, friendlyErrorMessage(err))))
    process.exit(1)
  }
}
