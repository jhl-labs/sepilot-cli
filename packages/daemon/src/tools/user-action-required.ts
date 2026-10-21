export class UserActionRequiredError extends Error {
  override name = 'UserActionRequiredError'

  constructor(
    message: string,
    readonly code?: 'browser_connection_required' | 'browser_control_required',
  ) {
    super(message)
  }
}
