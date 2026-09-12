/** Empty output has no partial answer to continue, even if reasoning consumed the budget. */
export function lengthRecoveryInstruction(partialOutput: string): string {
  return partialOutput.trim()
    ? 'Your previous reply was cut off because it reached the output length limit. Continue exactly from where you left off; do not repeat what you already wrote.'
    : 'The output length limit was reached without a usable answer or executable tool call. There is no partial answer to continue. Resume from the actual user request and retained tool evidence: take the next concrete tool step, or give a concise complete answer if no action is needed. Keep planning brief; do not restart the entire design or invent completed work.'
}
