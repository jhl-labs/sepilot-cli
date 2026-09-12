export async function formatChatStreamFailure(response: Response): Promise<string> {
  const body = await response.text().catch(() => '')
  return `chat stream failed (${response.status}): ${body || response.statusText}`
}
