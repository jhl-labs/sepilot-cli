import type React from 'react'
import { useCallback, useState } from 'react'

export interface UseSkillManagerResult {
  skillManagerOpen: boolean
  setSkillManagerOpen: React.Dispatch<React.SetStateAction<boolean>>
  skillManagerQuery: string
  setSkillManagerQuery: React.Dispatch<React.SetStateAction<string>>
  skillManagerIndex: number
  setSkillManagerIndex: React.Dispatch<React.SetStateAction<number>>
  skillManagerTogglingId: string | null
  setSkillManagerTogglingId: React.Dispatch<React.SetStateAction<string | null>>
  skillManagerError: string | null
  setSkillManagerError: React.Dispatch<React.SetStateAction<string | null>>
  skillManagerMessage: string | null
  setSkillManagerMessage: React.Dispatch<React.SetStateAction<string | null>>
  closeSkillManagerPicker: () => void
}

export function useSkillManager(): UseSkillManagerResult {
  const [skillManagerOpen, setSkillManagerOpen] = useState(false)
  const [skillManagerQuery, setSkillManagerQuery] = useState('')
  const [skillManagerIndex, setSkillManagerIndex] = useState(0)
  const [skillManagerTogglingId, setSkillManagerTogglingId] = useState<string | null>(null)
  const [skillManagerError, setSkillManagerError] = useState<string | null>(null)
  const [skillManagerMessage, setSkillManagerMessage] = useState<string | null>(null)

  const closeSkillManagerPicker = useCallback(() => {
    setSkillManagerOpen(false)
    setSkillManagerTogglingId(null)
    setSkillManagerError(null)
    setSkillManagerMessage(null)
  }, [])

  return {
    skillManagerOpen,
    setSkillManagerOpen,
    skillManagerQuery,
    setSkillManagerQuery,
    skillManagerIndex,
    setSkillManagerIndex,
    skillManagerTogglingId,
    setSkillManagerTogglingId,
    skillManagerError,
    setSkillManagerError,
    skillManagerMessage,
    setSkillManagerMessage,
    closeSkillManagerPicker,
  }
}
