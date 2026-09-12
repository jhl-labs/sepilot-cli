import type React from 'react'
import { ApprovalModal } from './ApprovalModal.js'
import { AttachmentPalette } from './AttachmentPalette.js'
import { AutonomyPicker } from './AutonomyPicker.js'
import { CommandPalette } from './CommandPalette.js'
import { FilePicker } from './FilePicker.js'
import { HelpModal } from './HelpModal.js'
import { ModePicker } from './ModePicker.js'
import { ModelPicker } from './ModelPicker.js'
import { ProviderDeleteModal } from './ProviderDeleteModal.js'
import { ProviderSetupModal } from './ProviderSetupModal.js'
import { SessionPicker } from './SessionPicker.js'
import { SkillManagerPicker } from './SkillManagerPicker.js'
import { SkillPalette } from './SkillPalette.js'
import { SkillStorePicker } from './SkillStorePicker.js'

interface OverlayPanelsProps {
  commandPalette?: React.ComponentProps<typeof CommandPalette> | null
  attachmentPalette?: React.ComponentProps<typeof AttachmentPalette> | null
  skillPalette?: React.ComponentProps<typeof SkillPalette> | null
  autonomyPicker?: React.ComponentProps<typeof AutonomyPicker> | null
  modelPicker?: React.ComponentProps<typeof ModelPicker> | null
  providerSetupModal?: React.ComponentProps<typeof ProviderSetupModal> | null
  providerDeleteModal?: React.ComponentProps<typeof ProviderDeleteModal> | null
  modePicker?: React.ComponentProps<typeof ModePicker> | null
  filePicker?: React.ComponentProps<typeof FilePicker> | null
  sessionPicker?: React.ComponentProps<typeof SessionPicker> | null
  skillManagerPicker?: React.ComponentProps<typeof SkillManagerPicker> | null
  skillStorePicker?: React.ComponentProps<typeof SkillStorePicker> | null
  approvalModal?: React.ComponentProps<typeof ApprovalModal> | null
  helpModal?: React.ComponentProps<typeof HelpModal> | null
}

export function OverlayPanels({
  commandPalette,
  attachmentPalette,
  skillPalette,
  autonomyPicker,
  modelPicker,
  providerSetupModal,
  providerDeleteModal,
  modePicker,
  filePicker,
  sessionPicker,
  skillManagerPicker,
  skillStorePicker,
  approvalModal,
  helpModal,
}: OverlayPanelsProps) {
  return (
    <>
      {commandPalette && <CommandPalette {...commandPalette} />}
      {attachmentPalette && <AttachmentPalette {...attachmentPalette} />}
      {skillPalette && <SkillPalette {...skillPalette} />}
      {autonomyPicker && <AutonomyPicker {...autonomyPicker} />}
      {modelPicker && <ModelPicker {...modelPicker} />}
      {providerSetupModal && <ProviderSetupModal {...providerSetupModal} />}
      {providerDeleteModal && <ProviderDeleteModal {...providerDeleteModal} />}
      {modePicker && <ModePicker {...modePicker} />}
      {filePicker && <FilePicker {...filePicker} />}
      {sessionPicker && <SessionPicker {...sessionPicker} />}
      {skillManagerPicker && <SkillManagerPicker {...skillManagerPicker} />}
      {skillStorePicker && <SkillStorePicker {...skillStorePicker} />}
      {approvalModal && <ApprovalModal {...approvalModal} />}
      {helpModal && <HelpModal {...helpModal} />}
    </>
  )
}
