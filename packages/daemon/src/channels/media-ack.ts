/**
 * Shared copy for acknowledging inbound media that the channel adapters cannot
 * yet process (images, audio, video, files). Adapters previously dropped such
 * messages with no signal, leaving the sender to wonder whether the bot was
 * broken. Acking is the minimal correct behavior until real media handling
 * (fetch + size limit + vision/transcription) lands.
 */
export const UNSUPPORTED_MEDIA_MESSAGE =
  'Sorry — I can only handle text messages right now. Media (images, audio, video, files) is not supported yet.'
