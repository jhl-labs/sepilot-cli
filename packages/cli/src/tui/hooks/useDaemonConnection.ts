// Three pieces of daemon-connection state App.tsx kept inline:
//
// - version — the daemon's reported version string (header badge)
// - connectionStatus — 'connecting' | 'connected' | 'error', drives
//   the splash screen / header dot
// - connectionError — the message to show under the splash on
//   connection failure
//
// All three are populated together by the bootstrap/heartbeat effects;
// bundling them keeps App.tsx from declaring three useStates in a row
// and signals that they always travel together.

import { useState } from 'react'

type DaemonConnectionStatus = 'connecting' | 'connected' | 'error'

export interface UseDaemonConnectionResult {
  version: string
  setVersion: React.Dispatch<React.SetStateAction<string>>
  connectionStatus: DaemonConnectionStatus
  setConnectionStatus: React.Dispatch<
    React.SetStateAction<DaemonConnectionStatus>
  >
  connectionError: string | null
  setConnectionError: React.Dispatch<React.SetStateAction<string | null>>
}

export function useDaemonConnection(): UseDaemonConnectionResult {
  const [version, setVersion] = useState('0.0.0')
  const [connectionStatus, setConnectionStatus] =
    useState<DaemonConnectionStatus>('connecting')
  const [connectionError, setConnectionError] = useState<string | null>(null)

  return {
    version,
    setVersion,
    connectionStatus,
    setConnectionStatus,
    connectionError,
    setConnectionError,
  }
}
