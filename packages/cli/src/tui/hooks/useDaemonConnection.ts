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
