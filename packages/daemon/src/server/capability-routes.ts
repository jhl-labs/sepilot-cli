import type { FastifyPluginAsync } from 'fastify'
import { registerCapabilitiesRoute } from './routes/capabilities.js'
import {
  registerNetworkCapabilityRoutes,
  defaultNetworkProbe,
} from './routes/network-capability.js'
import { registerSettingsJsonRoutes } from './routes/settings-json.js'
import { registerSystemBootInfoRoutes } from './routes/system-boot-info.js'
import { registerMessageQueueRoutes } from './routes/message-queue.js'
import { registerPersonaRoutes } from '../persona/routes.js'
import { registerSnippetsRoutes } from '../snippets/routes.js'
import { registerPromptTemplatesRoutes } from '../prompts/routes.js'
import { registerPersonalDocsRoutes } from '../personal-docs/routes.js'
import { registerQuickInputRoutes } from './routes/quick-input.js'
import { registerNotificationsCapabilityRoutes } from './routes/notifications-capability.js'
import { registerSchedulerCapabilityRoutes } from './routes/scheduler-capability.js'
import { registerBackupCapabilityRoutes } from './routes/backup-capability.js'
import { registerGitHubChatSyncRoutes } from './routes/github-chat-sync.js'
import { registerGitHubOAuthRoutes } from './routes/github-oauth.js'
import { registerGitHubProjectsRoutes } from './routes/github-projects.js'
import { registerGitHubSyncRoutes } from './routes/github-sync.js'
import { registerTeamDocsRoutes } from '../team-docs/routes.js'
import { registerMessageSubscriptionRoutes } from '../message-subscription/routes.js'
import { registerUsageSnapshotRoute } from './routes/usage-snapshot.js'
import { registerLlmProvidersCapabilityRoutes } from './routes/llm-providers-capability.js'
import { registerSettingsCapabilityRoutes } from './routes/settings-capability.js'
import { registerSkillsCapabilityRoutes } from './routes/skills-capability.js'
import { registerRagCapabilityRoutes } from './routes/rag-capability.js'
import { registerImageGenCapabilityRoutes } from './routes/image-gen-capability.js'
import { registerMcpCapabilityRoutes } from './routes/mcp-capability.js'
import { getImageGenQueue, imageGenProviders } from '../media/image-gen/runtime.js'
import {
  registerWikiFeatureRoutes,
  registerExtensionsFeatureRoutes,
} from '../generated/feature-registration.js'

/**
 * Phase D0–D6 capability routes (`/capabilities`, `/persona`, `/snippets`,
 * `/prompt-templates`, `/personal-docs`, `/wiki/*`, `/network`, `/settings/json`,
 * `/message-queue/snapshot`, `/quick-input/publish`, `/notifications*`,
 * `/scheduler/jobs`, `/backup`, `/github/*`, `/mcp/*`).
 *
 * desktop UI/UX parity 시리즈가 약속한 thin surface ↔ daemon 계약면.
 * 기존 `/api/v1/*` 라우트와 경로 네임스페이스가 분리되어 있어 충돌 없이
 * 병렬로 노출된다.
 */
export const capabilityRoutes: FastifyPluginAsync = async (app) => {
  await registerCapabilitiesRoute(app)
  await registerNetworkCapabilityRoutes(app, defaultNetworkProbe)
  await registerSettingsJsonRoutes(app)
  await registerSystemBootInfoRoutes(app)
  await registerMessageQueueRoutes(app)
  await registerPersonaRoutes(app)
  await registerSnippetsRoutes(app)
  await registerPromptTemplatesRoutes(app)
  await registerPersonalDocsRoutes(app)
  await registerWikiFeatureRoutes(app)
  await registerQuickInputRoutes(app)
  await registerNotificationsCapabilityRoutes(app)
  await registerSchedulerCapabilityRoutes(app)
  await registerBackupCapabilityRoutes(app)
  await registerGitHubOAuthRoutes(app)
  await registerGitHubProjectsRoutes(app)
  await registerGitHubSyncRoutes(app)
  await registerGitHubChatSyncRoutes(app)
  await registerTeamDocsRoutes(app)
  await registerMessageSubscriptionRoutes(app)
  await registerExtensionsFeatureRoutes(app)
  await registerUsageSnapshotRoute(app)
  await registerLlmProvidersCapabilityRoutes(app)
  await registerSettingsCapabilityRoutes(app)
  await registerSkillsCapabilityRoutes(app)
  await registerRagCapabilityRoutes(app)
  await registerMcpCapabilityRoutes(app)
  await registerImageGenCapabilityRoutes(app, getImageGenQueue(), imageGenProviders)
}
