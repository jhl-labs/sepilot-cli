import type { Disposable } from '../disposable.js'
import type { HookEvent, HookPayload, HookResult, IHookHandler } from './types.js'

export interface IHookRegistry {
  register(event: HookEvent, handler: IHookHandler): Disposable
  trigger(payload: HookPayload, signal?: AbortSignal): Promise<HookResult>
}
