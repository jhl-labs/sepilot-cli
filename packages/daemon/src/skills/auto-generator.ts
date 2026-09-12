import type { ILLMProvider, SessionEvent, SkillMetadata } from '@sepilotd/core'

export interface GeneratedSkill {
  metadata: SkillMetadata
  content: string
}

export async function generateSkillFromSession(
  provider: ILLMProvider,
  events: SessionEvent[],
  model: string,
): Promise<GeneratedSkill | null> {
  // Extract tool calls and results
  const toolEvents = events.filter(e => e.type === 'tool_call' || e.type === 'tool_result')
  if (toolEvents.length < 2) return null // Not enough tool usage to make a skill

  const messages = events
    .filter(e => e.type === 'user_message' || e.type === 'assistant_message')
    .map((e) => `[${e.type}] ${(e as { content: string }).content}`)
    .join('\n')

  const tools = [...new Set(toolEvents.filter(e => e.type === 'tool_call').map((e) => (e as { tool: string }).tool))]

  const prompt = `Analyze this conversation and generate a reusable skill definition.

Conversation:
${messages.slice(0, 5000)}

Tools used: ${tools.join(', ')}

Generate a SKILL.md with TOML frontmatter (+++). Include:
- name (kebab-case)
- version "1.0.0"
- description (one line)
- tools array
- Steps section explaining how to reproduce this task

Return ONLY the SKILL.md content, starting with +++.`

  try {
    const response = await provider.chat({
      model,
      messages: [{ role: 'user', content: prompt }],
      maxTokens: 2000,
    })

    const content = typeof response.message.content === 'string' ? response.message.content : ''
    if (!content.includes('+++')) return null

    // Extract metadata from generated content
    const match = content.match(/\+\+\+\n([\s\S]*?)\n\+\+\+/)
    if (!match) return null

    const id = `auto-${Date.now()}`
    return {
      metadata: {
        id,
        name: id,
        version: '1.0.0',
        description: 'Auto-generated skill',
        tools,
        author: 'auto-generated',
        created: new Date().toISOString().split('T')[0],
      },
      content,
    }
  } catch {
    return null
  }
}
