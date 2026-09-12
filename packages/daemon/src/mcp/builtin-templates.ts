import type { McpServerTemplate } from './marketplace-source.js'

export const BUILTIN_MCP_MARKETPLACE = 'builtin'

export const builtinMcpServerTemplates: McpServerTemplate[] = [
  {
    name: 'notion',
    description:
      'Notion workspace search, page/database updates, docs, tasks, and project notes through the official hosted Notion MCP server.',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', 'mcp-remote', 'https://mcp.notion.com/mcp'],
    env: {},
    tags: ['notion', 'docs', 'knowledge', 'tasks', 'workspace'],
    homepage: 'https://developers.notion.com/guides/mcp/mcp',
    marketplace: BUILTIN_MCP_MARKETPLACE,
  },
  {
    name: 'github',
    description:
      'GitHub repositories, issues, pull requests, code search, and workflow context through GitHub MCP.',
    transport: 'http',
    url: 'https://api.githubcopilot.com/mcp/',
    headers: {
      Authorization: 'Bearer {secret.GITHUB_TOKEN}',
    },
    tags: ['github', 'git', 'issues', 'pull-requests', 'development'],
    homepage: 'https://github.com/github/github-mcp-server',
    marketplace: BUILTIN_MCP_MARKETPLACE,
  },
  {
    name: 'linear',
    description:
      'Linear issue, project, cycle, and comment workflows through Linear remote MCP.',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', 'mcp-remote', 'https://mcp.linear.app/mcp'],
    env: {},
    tags: ['linear', 'issues', 'projects', 'product', 'planning'],
    homepage: 'https://linear.app/docs/mcp',
    marketplace: BUILTIN_MCP_MARKETPLACE,
  },
  {
    name: 'atlassian',
    description:
      'Jira, Confluence, Compass, and Atlassian project knowledge through Atlassian Rovo MCP.',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', 'mcp-remote', 'https://mcp.atlassian.com/v1/mcp/authv2'],
    env: {},
    tags: ['atlassian', 'jira', 'confluence', 'compass', 'planning'],
    homepage:
      'https://support.atlassian.com/atlassian-rovo-mcp-server/docs/getting-started-with-the-atlassian-remote-mcp-server/',
    marketplace: BUILTIN_MCP_MARKETPLACE,
  },
  {
    name: 'slack-mcp',
    description:
      'Slack search, channel history, messages, canvases, and user context through Slack MCP. Requires a Slack-approved MCP client/app setup.',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', 'mcp-remote', 'https://mcp.slack.com/mcp'],
    env: {},
    tags: ['slack', 'chat', 'messages', 'canvases', 'team'],
    homepage: 'https://docs.slack.dev/ai/mcp-server/',
    marketplace: BUILTIN_MCP_MARKETPLACE,
  },
  {
    name: 'google-workspace-remote',
    description:
      'Bridge to a reviewed Google Workspace MCP endpoint for Gmail, Calendar, Drive, Docs, Sheets, and Slides. Provide GOOGLE_WORKSPACE_MCP_URL during install.',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', 'mcp-remote', '{input.GOOGLE_WORKSPACE_MCP_URL}'],
    env: {},
    tags: ['google', 'gmail', 'email', 'calendar', 'drive', 'workspace'],
    homepage: 'https://github.com/google/mcp',
    variables: [
      {
        name: 'GOOGLE_WORKSPACE_MCP_URL',
        label: 'Google Workspace MCP URL',
        description: 'Team-approved remote MCP endpoint URL.',
        placeholder: 'https://...',
        required: true,
      },
    ],
    marketplace: BUILTIN_MCP_MARKETPLACE,
  },
  {
    name: 'email-remote',
    description:
      'Bridge to a reviewed Gmail, Outlook, IMAP, or SMTP MCP endpoint. Provide EMAIL_MCP_URL during install.',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', 'mcp-remote', '{input.EMAIL_MCP_URL}'],
    env: {},
    tags: ['email', 'gmail', 'outlook', 'imap', 'smtp'],
    homepage: 'https://modelcontextprotocol.io/registry/about',
    variables: [
      {
        name: 'EMAIL_MCP_URL',
        label: 'Email MCP URL',
        description: 'Remote MCP endpoint for Gmail, Outlook, IMAP, or SMTP.',
        placeholder: 'https://...',
        required: true,
      },
    ],
    marketplace: BUILTIN_MCP_MARKETPLACE,
  },
  {
    name: 'zapier-remote',
    description:
      'Bridge to a Zapier MCP endpoint for app automation across CRM, email, calendar, project management, and SaaS tools. Provide ZAPIER_MCP_URL during install.',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', 'mcp-remote', '{input.ZAPIER_MCP_URL}'],
    env: {},
    tags: ['zapier', 'automation', 'saas', 'crm', 'email'],
    homepage: 'https://modelcontextprotocol.io/registry/about',
    variables: [
      {
        name: 'ZAPIER_MCP_URL',
        label: 'Zapier MCP URL',
        description: 'Zapier MCP endpoint URL for the target automation account.',
        placeholder: 'https://...',
        required: true,
      },
    ],
    marketplace: BUILTIN_MCP_MARKETPLACE,
  },
]
