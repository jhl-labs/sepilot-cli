import type { ScheduledJob } from './job-store.js'
import {
  ALL_NOTIFICATION_AUDIENCES,
  normalizeNotificationAudience,
  updateNotificationAudienceSubscription,
  type NotificationAudience,
} from '../notifications/audience.js'

const SCHEDULER_NOTIFICATION_SUBSCRIBERS_KEY = 'notificationSubscribers'

export interface SchedulerNotificationSubscriptions {
  jobId: string
  subscribers: string[]
}

export function getSchedulerNotificationSubscribers(job: ScheduledJob): string[] {
  const metadata = job.metadata ?? {}
  const subscribers = normalizeNotificationAudience(
    metadata[SCHEDULER_NOTIFICATION_SUBSCRIBERS_KEY],
  )
  return subscribers ?? [ALL_NOTIFICATION_AUDIENCES]
}

export function schedulerNotificationAudience(job: ScheduledJob): NotificationAudience {
  const subscribers = getSchedulerNotificationSubscribers(job)
  if (
    subscribers.length === 1
    && subscribers[0] === ALL_NOTIFICATION_AUDIENCES
  ) {
    return null
  }
  return subscribers
}

export function setSchedulerNotificationSubscribers(
  metadata: Record<string, unknown> | null | undefined,
  subscribers: unknown,
): Record<string, unknown> {
  const normalized = normalizeNotificationAudience(subscribers) ?? [ALL_NOTIFICATION_AUDIENCES]
  return {
    ...(metadata ?? {}),
    [SCHEDULER_NOTIFICATION_SUBSCRIBERS_KEY]: normalized,
  }
}

export function updateSchedulerNotificationSubscription(
  metadata: Record<string, unknown> | null | undefined,
  surface: string,
  subscribed: boolean,
): Record<string, unknown> {
  const current = normalizeNotificationAudience(
    metadata?.[SCHEDULER_NOTIFICATION_SUBSCRIBERS_KEY],
  )
  return setSchedulerNotificationSubscribers(
    metadata,
    updateNotificationAudienceSubscription(current, surface, subscribed),
  )
}

export function describeSchedulerNotificationSubscriptions(
  job: ScheduledJob,
): SchedulerNotificationSubscriptions {
  return {
    jobId: job.id,
    subscribers: getSchedulerNotificationSubscribers(job),
  }
}
