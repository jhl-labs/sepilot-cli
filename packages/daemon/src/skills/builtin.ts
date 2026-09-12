import { TASK_INVOCATION_GUIDANCE, TASK_MANAGEMENT_TOOLS } from '../agent/task-invocation.js'
import { AutonomyLevel } from '@sepilotd/core'
import type { SkillMetadata } from '@sepilotd/core'
import { isDeepStrictEqual } from 'node:util'
import type { FileSkillRegistry } from './registry.js'

export interface BuiltinSkill {
  metadata: SkillMetadata
  content: string
}

export const builtinSkills: BuiltinSkill[] = [
  {
      metadata: {
        id: 'task', name: 'task', version: '1.0.1', author: 'sepilotd',
        description: 'Manage durable Tasks: explicitly requested future, recurring, or background agent work, and existing task status, results, edits, pause, resume, or cancellation. Also available with a leading $task command. Not for ordinary synchronous work, dates mentioned as context, or explanations of task scheduling.',
        tags: ['builtin', 'tasks', 'scheduler'], tools: [...TASK_MANAGEMENT_TOOLS],
        autonomy_required: AutonomyLevel.Supervised, enabled: true,
      },
      content: TASK_INVOCATION_GUIDANCE,
    },
  {
      metadata: {
        id: 'skill-author',
        name: 'skill-author',
        version: '1.1.0',
        description:
          'Create, update, or extract agent skills from repeated workflows with validation and approval guardrails.',
        author: 'sepilotd',
        tags: ['builtin', 'skills', 'authoring'],
        tools: ['skill', 'fs.read', 'fs.write'],
        autonomy_required: AutonomyLevel.Supervised,
        enabled: true,
      },
      content: `# Skill Author
  
  Use this skill when the user asks to create, update, review, or install an agent skill, or when they ask whether a repeated workflow should become a skill.
  
  ## Workflow
  
  1. Identify the workflow the skill should capture. Prefer a skill only when the task is repeated, procedural, domain-specific, or depends on repo/local conventions.
  2. Check for existing related skills before drafting. If a related skill exists, update or extend it instead of creating a duplicate.
  3. Draft a concise SKILL.md:
     - Frontmatter must include name and description.
     - Description must state when the skill should trigger.
     - Body should contain only the procedure the agent needs.
     - Move large references into references/ and mention exactly when to read them.
     - Use scripts/ only for deterministic or repeatedly rewritten operations.
  4. Keep the skill safe:
     - Do not embed secrets, personal data, tokens, or machine-specific absolute paths.
     - Declare only tools the skill truly needs.
     - State autonomy or approval requirements for destructive, network, or credentialed operations.
     - Do not create extra README, changelog, install guide, or other auxiliary docs.
  5. Validate before persistence:
     - Parse frontmatter.
     - Check required metadata.
     - Check declared tools exist when the runtime exposes a validator.
     - Search for obvious dangerous shell patterns.
     - Run a small dry-run against one realistic prompt when feasible.
  6. Persist only when appropriate:
     - If the user explicitly asked to create or update the skill, write it through the daemon skill API or the repository change requested by the user.
     - If the user did not explicitly approve persistence, present the draft and ask for approval.
     - If the platform supports draft or disabled skills, prefer draft first. If it does not, avoid installing until approval is clear.
  
  ## Output Shape
  
  When proposing a new skill, provide:
  
  - Skill id
  - Trigger description
  - Required tools
  - Main guardrails
  - The SKILL.md content or the exact files changed
  - Validation performed and any remaining risk
  
  When modifying an existing skill, keep unrelated wording and bundled resources intact unless the requested change requires them.
  `,
    },
  {
      metadata: {
        id: 'monitor-infrastructure',
        name: 'monitor-infrastructure',
        version: '1.6.3',
        description:
          'Inspect and continuously monitor host, HTTP service, Kubernetes infrastructure, and bounded Gitea Actions run state; establish a read-only baseline, define explicit thresholds, schedule recurring resource digests or transition-only anomaly checks, and deliver authorized alerts through Sepilot notifications/Mattermost. Trigger for infrastructure health checks, uptime monitoring, CPU/memory/disk reports, pod or node monitoring, CI run status, anomaly alerts, and recurring operational status reports. Do not use for applying infrastructure or CI changes.',
        author: 'sepilotd',
        tags: ['builtin', 'personal', 'monitoring', 'infrastructure', 'sre', 'scheduler', 'alerts'],
        tools: [
          'assistant.status',
          'gitea.actions.runs.inspect',
          'system.info',
          'terminal.run',
          'webfetch',
          'schedule_create',
          'schedule_list',
          'schedule_get',
          'schedule_update',
          'schedule_pause',
          'schedule_resume',
          'schedule_cancel',
          'schedule_runs',
          'schedule_run_now',
          'monitor.evaluate',
          'monitor.report',
          'notification.publish',
        ],
        autonomy_required: AutonomyLevel.Supervised,
        enabled: true,
      },
      content: `# Infrastructure Monitor
  
  Build monitoring from explicit targets and structured tool contracts. Do not route by matching one request phrase, and do not mutate infrastructure.
  
  ## Boundaries
  
  - Limit probes to read-only host metrics, HTTP(S) health endpoints, read-only \`kubectl\` commands, and bounded \`tea actions runs list/view\` metadata for the active workspace repository.
  - Never read or print Secret data, kubeconfig or tea credentials, environment dumps, authorization headers, action logs, action secrets/variables, or response fields known to contain credentials.
  - Treat a timeout, permission failure, missing metrics API, invalid response, and partial target coverage as \`unknown\` or \`error\`, never \`healthy\`.
  - Do not schedule anything until the user has supplied or accepted the target, cadence, thresholds, timezone, and destinations.
  - Do not set \`unattended: true\` unless the user explicitly authorizes future runs without interactive approvals.
  - \`unattended\` is standing approval authority, not the scheduler enable switch. \`unattended: false\` does not mean that a recurring job will not fire: use \`enabled\`, \`status\`, \`nextRunAt\`, and \`schedule_runs\` as firing evidence. A false value means only that a future policy-gated tool call may still require interactive approval; autonomously allowed read-only calls can run on schedule.
  - Do not apply, patch, restart, scale, delete, exec, or port-forward. Hand an approved remediation request to the \`kubectl\` skill instead.
  
  When the active turn is ReadOnly, use only the observation lane: \`assistant.status\`, \`system.info\`, \`webfetch\`, direct audited \`kubectl\` or Gitea Actions run inspection, schedule list/get/runs, and \`monitor.report\`. In that lane, do not call \`schedule_create\` or any update/pause/resume/cancel/run-now tool, and do not call \`notification.publish\`. A one-off baseline without an accepted monitor contract must not call \`monitor.evaluate\` merely to persist a sample. If the user also requested scheduling or delivery, finish the available baseline, state exactly which mutation was not performed because of the ReadOnly ceiling, and ask whether they want a supervised turn; do not silently drop the selected skill or claim that delivery was configured.
  
  ## Define the monitor
  
  Record a compact contract before probing:
  
  - Stable monitor id and target kind: \`assistant-runtime\`, \`host\`, \`http\`, \`kubernetes\`, or \`gitea-actions\`.
  - Exact host scope, URL, or Kubernetes context + namespace + resource selector.
  - Read-only probes and per-probe timeout.
  - Thresholds and required consecutive anomaly, recovery, and observation-error samples. Do not invent thresholds when the user has not accepted defaults.
  - Evaluation cadence, digest cadence, timezone, and notification destinations.
  - Severity for each condition and the recovery rule.
  
  For daemon-owned assistant readiness, configured channel state, notification/Relay delivery state, or scheduler inventory, use \`assistant.status\` with only the requested sections and exact \`channelTypes\` when a channel such as Mattermost is named. Treat that structured runtime projection as the authority; do not search repository source or configuration files and do not call the daemon's HTTP API through \`terminal.run\` or \`webfetch\`. A missing requested channel is returned explicitly as \`not_configured\`, and an absent or failed snapshot is \`unknown\`.
  
  For Kubernetes, always pass \`--context\` and \`--namespace\` after the target is known. Bound logs/events and avoid Secret output. Do not guess shortened resource names from product labels. When Deployment readiness and image are requested, prefer one bounded inventory projection instead of full JSON: \`kubectl --context <context> --namespace <namespace> get deployments -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas,IMAGE:.spec.template.spec.containers[*].image --no-headers\`. Select the requested rows from that result and do not add namespace/context discovery after the target is already established. For host metrics, use \`system.info\`; for an HTTP target, use \`webfetch\`; use \`terminal.run\` only for bounded read-only Kubernetes queries or one \`git remote\` name inventory. Invoke \`kubectl\` or \`git\` directly as the \`terminal.run\` executable with one structured argument array and one probe per call. Do not wrap either in \`bash -c\`, \`sh -c\`, or another shell and do not join multiple probes into one command.
  
  For Gitea CI, \`<workspace-remote>\` means a literal local alias returned by \`git remote\` (normally \`origin\`), never the remote URL. If the alias is not already known, inspect names only with \`git remote\`; prefer an existing \`origin\`, and report \`unknown\` instead of guessing when the alias is ambiguous. Call \`gitea.actions.runs.inspect\` once with that exact alias and a bounded limit. Inside that one canonical call, the daemon enriches at most five priority runs with conclusion, timestamps, duration, and a validated run URL; actual \`in_progress\` or \`running\` runs are prioritized and may include bounded jobs while reusing the alias byte-for-byte; do not replace this canonical tool with separate \`tea\` terminal calls. Treat every null field and each run named in \`runDetails.failures\` as \`unknown\`; when \`runDetails.complete=false\`, do not generalize detail coverage beyond non-null per-row fields. Preserve exact status and conclusion categories: only \`failure\` is a failed run; \`cancelled\`, \`skipped\`, \`queued\`, and \`in_progress\` remain distinct. The daemon normalizes zero/uninitialized timestamps and their derived durations to \`null\`; report those values as \`unknown\` and never compute age or duration from them. Keep the active workspace remote as repository authority. Action logs, secrets/variables, repository/login overrides, workflow management, cancellation, deletion, and reruns are outside the tool. Missing binary/config, authorization failure, malformed/unsupported output, mismatched run identity, or incomplete detail expansion is \`unknown\`, not a healthy CI claim.
  
  ## Baseline first
  
  1. Run one current read-only probe.
  2. Report the observation timestamp, target, evidence values, missing signals, and status.
  3. When the user has accepted a monitor contract and thresholds, compare the evidence and call \`monitor.evaluate\` exactly once with a stable non-secret sample id. For a one-off status request without that contract, report the observation without persisting a synthetic monitor sample. Use \`unknown\` for every incomplete or failed observation; never convert it to \`healthy\`.
  4. Ask for any missing schedule or unattended authorization before creating jobs.
  
  Report readiness at the layer actually observed. Deployment replica readiness proves only that the workload controller has ready replicas; it does not prove that a Service, Endpoint/EndpointSlice, Ingress, health route, or end-to-end traffic path works. An aggregate \`Ready\` verdict is allowed only when current-turn evidence covers every layer necessary for the claimed capability. Otherwise use a narrower label such as \`workload ready; service path UNVERIFIED\` or an overall partial/unknown verdict, and name the smallest missing read-only probe.
  
  ## Schedule shape
  
  Prefer two independent recurring jobs when the user wants both immediate anomalies and a quieter resource report:
  
  1. **Transition checker:** run at the evaluation cadence. Give \`schedule_create\` a self-contained instruction containing the entire monitor contract, set \`skill_refs: [{"name":"monitor-infrastructure"}]\`, set \`notification_priority: high\` (or \`critical\` only when the accepted severity demands it), and use retry settings appropriate to the target. After one probe, call \`monitor.evaluate\` with the accepted contract version and thresholds, a unique sample id derived from the scheduler run, a bounded redacted evidence summary, and numeric metrics when available. Trust its structured transition result: emit final text only when \`shouldNotify=true\`; return no visible text for \`quiet\`, \`pending\`, \`duplicate\`, \`stale\`, and non-notifying observation recovery. Never reconstruct or override consecutive counts in prose. A retry must reuse the same sample id, and must not send a second incident.
  2. **Digest:** run at the requested report cadence with \`skill_refs: [{"name":"monitor-infrastructure"}]\` and \`notification_priority: normal\`. Call \`monitor.report\` for the accepted lookback and produce its bounded summary: period, availability/sample count, current/min/max/average resource evidence, anomalies/recoveries, unknown intervals, retention limits, and next run. Do not claim history outside its retained evidence.
  
  Before creating either job, call \`schedule_list\` and reconcile equivalent jobs. Never duplicate a monitor because the user paraphrased the request. When the user changes its name, cadence, instruction, retry policy, notification priority, explicit unattended authority, or delivery destination, use \`schedule_update\` so the job identity and run history survive; omitted fields retain their current values, and cancel/recreate is only for changing the requested one-shot/recurring kind. A destination is an atomic \`channel_type\`/\`channel_target\` pair with an optional \`reply_to_message_id\`, and \`clear_delivery_route: true\` removes all three. Never infer \`unattended: true\` or a destination the user did not authorize. After a successful \`schedule_create\`, follow its atomicity contract: confirm the schedule and stop the turn.
  
  Use \`notification.publish\` only for an explicitly authorized immediate interactive alert outside the scheduler path. Reuse \`monitor.evaluate.notificationDedupKey\`; do not invent a second event identity. Scheduled job output already reaches its originating channel and relay, so do not also call \`notification.publish\` for the same scheduled transition.
  
  ## Final report
  
  - Monitor id, target, probes, thresholds, and baseline evidence.
  - Created or reused schedule ids, enabled/status/next run, timezone, and retry settings; report job-scoped standing approval separately from scheduled firing.
  - Alert/digest delivery behavior and configured relay availability if known.
  - Unknown signals, limitations, and the exact read-only next check.
  `,
    },
  {
      metadata: {
        id: 'research-to-jpad',
        name: 'research-to-jpad',
        version: '1.0.6',
        description:
          'Research a question through configured public web search or the internal AI Search provider, preserve source provenance, synthesize a citation-backed Markdown draft, and create or update a JPAD page only after explicit destination and final-content confirmation. Trigger when the user asks to investigate, analyze, and record or publish the result in JPAD. Do not use for research that the user did not ask to publish.',
        author: 'sepilotd',
        tags: ['builtin', 'personal', 'research', 'internal-search', 'jpad', 'publishing'],
        tools: [
          'web.search',
          'webfetch',
          'fs.write',
          'jpad.workspaces',
          'jpad.pages.list',
          'jpad.pages.get',
          'jpad.pages.create',
          'jpad.pages.update',
          'memory.documents.ingest',
        ],
        autonomy_required: AutonomyLevel.Supervised,
        execution: {
          maxCompletionRetries: 2,
          stages: [
            {
              id: 'collect-research-sources',
              tools: ['web.search'],
              requiredForCompletion: true,
              satisfyOn: 'executed-outcome',
            },
            {
              id: 'resolve-jpad-workspace',
              tools: ['jpad.workspaces'],
              maxCallsPerTurn: 1,
              requires: ['collect-research-sources'],
              requiredForCompletion: true,
            },
            {
              id: 'inspect-jpad-pages',
              tools: ['jpad.pages.list'],
              requires: ['resolve-jpad-workspace'],
              requiredForCompletion: true,
            },
          ],
        },
        enabled: true,
      },
      content: `# Research to JPAD
  
  Produce an auditable research draft, then publish only the user-approved version and destination.
  
  When the active turn is ReadOnly, complete the research and JPAD discovery/read lane, but do not call \`jpad.pages.create\` or \`jpad.pages.update\`, and do not call \`fs.write\` or \`memory.documents.ingest\`. Return the complete draft in the final response, identify the observed destination candidate, and say \`not published: ReadOnly\`. If the request also asked to publish, ask whether the user wants a supervised turn after showing the draft; never discard the selected skill, omit the research, or claim that a page was recorded.
  
  ## Research
  
  1. Restate the question, scope, freshness requirement, and intended JPAD audience.
  2. Choose the requested source boundary in the \`web.search\` call. Set \`sourceKind: public-web\` when public or official websites are required, and \`sourceKind: internal-index\` when the user requests the configured internal/search-dev knowledge index. Omit it only when the configured provider is acceptable. Never silently substitute one boundary for the other.
  3. Preserve each result's URL and provenance fields; an \`ai-search\` result may carry internal source, domain, tier, type, and publication metadata. Never present an internal result as public-web corroboration.
  4. Fetch primary sources when available. Let claim coverage determine how many fetches are useful; do not impose an arbitrary numeric fetch cap. Cross-check volatile or decision-critical claims with independent sources.
  5. Ground every decision-critical factual claim in text actually observed in a successful search/fetch result. For each such claim, retain the direct source URL and a short supporting excerpt or an explicit evidence-map reference. A result title, endpoint name, compatibility knowledge, or plausible field name is not evidence that the source states the claim. If the supporting text was not observed, label the claim unverified or omit it; never fill schema fields, enum values, timestamps, or semantics from model inference.
  6. Distinguish source facts, model inference, and unknowns. A cited URL proves provenance only; it does not by itself prove every nearby sentence.
  7. Draft Markdown with title, date, executive summary, findings, evidence/citations, uncertainties, and methodology. Keep quotations short and retain direct links.
  8. If workspace writing is available, save the draft before any external publish attempt so a JPAD failure does not lose the work. Write only to the user-requested path or the active workspace path agreed with the user; if the boundary rejects that target, do not substitute another directory as if it were equivalent.
  
  ## Resolve the JPAD destination
  
  1. Call \`jpad.workspaces\`; do not guess a workspace id.
  2. Call \`jpad.pages.list\` for the chosen workspace and search for an existing matching page before creating one.
     When the user asks for an inventory spanning several returned workspaces, call it once for each workspace in that requested scope. Stop when every requested workspace has one current observation; do not impose a one-workspace cap and do not repeat an unchanged workspace merely to increase call count.
  3. If updating, call \`jpad.pages.get\` and reconcile the current page. Preserve its returned ETag for \`ifMatch\`.
  4. Show the user the workspace, parent/page, title, and complete final Markdown. Ask for explicit publish approval. Research approval is not publish approval.
  
  ## Publish
  
  - Create with \`jpad.pages.create\` only after confirmation; set \`confirmPublish: true\` as an attestation of that confirmation.
  - Update with \`jpad.pages.update\` only after confirmation and pass the exact ETag from the latest same-session read. One update attempt consumes that read evidence.
  - On a revision conflict, read again, reconcile, show the revised final Markdown, and obtain fresh approval. Never overwrite concurrent edits blindly.
  - After any ambiguous or failed update attempt, read again before retrying; do not reuse the old ETag or approval.
  - If a create response is lost, list pages again before retrying so a duplicate is not created.
  - Never put the JPAD token in tool input, Markdown, files, logs, or chat. The daemon reads it only from runtime configuration.
  - If publishing fails, report the failure and retain the draft. Never claim a page was recorded without a successful tool result containing page evidence.
  
  ## Final report
  
  - Question, source counts by public/internal provenance, and unresolved claims.
  - Draft path when one was written.
  - JPAD workspace/page id or URL and create/update outcome, or \`not published\` with the reason.
  `,
    },
  {
      metadata: {
        id: 'presentation-review',
        name: 'presentation-review',
        version: '1.3.1',
        description:
          'Review and discuss an existing Microsoft PowerPoint (.pptx) presentation one slide at a time in read-only desktop PowerPoint. Opens only workspace-contained .pptx files, reads the current slide text, speaker notes, and shape metadata, and pairs that evidence with a workspace-confined rendering of exactly that slide. Trigger for "review this deck", "read this pptx", slide-by-slide discussion, or next/previous-slide requests. Creation and edits belong to pptx-author after separate explicit approval.',
        author: 'sepilotd',
        tags: ['builtin', 'office', 'pptx', 'presentation', 'review', 'read-only'],
        tools: [
          'fs.glob',
          'office.open_presentation',
          'office.navigate_slide',
          'office.read_slide',
          'office.capture_slide',
        ],
        autonomy_required: AutonomyLevel.Supervised,
        execution: {
          maxCompletionRetries: 1,
          stages: [
            {
              id: 'resolve',
              tools: ['fs.glob'],
              maxCallsPerTurn: 1,
            },
            {
              id: 'open',
              tools: ['office.open_presentation'],
              maxCallsPerTurn: 1,
            },
            {
              id: 'navigate',
              tools: ['office.navigate_slide'],
              maxCallsPerTurn: 1,
            },
            {
              id: 'read',
              tools: ['office.read_slide'],
              maxCallsPerTurn: 1,
              requiredForCompletion: true,
            },
            {
              id: 'capture',
              tools: ['office.capture_slide'],
              maxCallsPerTurn: 1,
              requires: ['read'],
            },
          ],
          argumentBindings: [{
            id: 'slide-index',
            targets: [
              { stage: 'navigate', argument: 'index' },
              { stage: 'read', argument: 'index' },
              { stage: 'capture', argument: 'index' },
            ],
            allowMissing: true,
          }],
        },
        enabled: true,
      },
      content: `# Presentation Review
  
  Use this skill to review an existing workspace PowerPoint deck with the user, exactly one slide per conversational turn. This is a read-only review workflow, not an authoring workflow.
  
  ## Non-negotiable boundaries
  
  - Open only a \`.pptx\` inside the selected workspace with \`office.open_presentation\`. Never pass an outside-workspace override and never use a terminal, script, or alternate Office command to bypass a boundary failure.
  - Never call any edit, save, write, keyboard, mouse, or terminal tool from this skill. The declared tool list is the entire allowlist.
  - Do not review the whole deck in one response. Claims about narrative coherence across the deck must wait until the relevant slides have actually been reviewed.
  - If the user asks to modify the deck, describe the proposed change and ask for separate explicit approval to hand off to \`pptx-author\`. Do not edit during the review turn.
  
  ## First review turn
  
  1. Resolve the exact \`.pptx\` path. Use the user's exact path when supplied; use \`fs.glob\` only when the path is ambiguous.
  2. Call \`office.open_presentation\` with that path. It must report \`readOnly: true\`.
  3. Call \`office.read_slide\` for slide 1 (or the exact slide the user requested) using the same path.
  4. Call \`office.capture_slide\` for the same slide and path. It renders only that workspace slide and attaches the PNG for visual review without exposing another desktop window.
  5. Review only that slide, combining exact Office data with the rendered slide image. If capture is unavailable or the selected model rejects image input, state that limitation in one short sentence and review only the first successful structured slide result. Do not retry capture, read, navigation, model discovery, or file search to work around a known vision limitation.
  
  ## Navigation turns
  
  - "Next" means exactly one \`office.navigate_slide\` call with \`direction: "next"\`, followed by \`office.read_slide\` and \`office.capture_slide\` with no index. Never prefetch slide N+2.
  - "Previous" means exactly one navigation with \`direction: "previous"\`, then read and capture.
  - For an exact slide number, navigate once with \`direction: "index"\` and that 1-based index, then read and capture.
  - Reuse the same presentation path on every navigation and read call. Stop and tell the user if PowerPoint reports that the active file changed or the requested index is outside 1..\`slideCount\`.
  
  ## Per-slide response
  
  Keep the response compact and grounded in the current slide. Stay below 1,800 characters, use at most eight bullets total, and never repeat raw coordinates or metadata unless one value materially supports a finding:
  
  - **Slide N / total — title or purpose**
  - **What works** — message hierarchy, architecture logic, visual hierarchy, and evidence actually present.
  - **Risks / trade-offs** — ambiguity, missing assumptions, technical risk, or audience mismatch visible on this slide.
  - **Discussion question** — one or two focused questions that help the user decide what to do next.
  
  Mention speaker notes when they materially change the interpretation. Distinguish visible evidence from inference. End by inviting the user to say "next slide" / "다음 슬라이드", "previous slide" / "이전 슬라이드", an exact slide number, or ask a question about the current slide. Keeping the word "slide" in short navigation turns lets automatic skill routing preserve this review workflow.
  `,
    },
  {
      metadata: {
        id: 'pptx-author',
        name: 'pptx-author',
        version: '1.3.0',
        description:
          'Author or edit Microsoft PowerPoint (.pptx) decks via python-pptx (or Marp / reveal-md for code-driven alternatives). Title + content slides with bullets, two-column, image, code, table, and chart layouts; speaker notes; theme application; section dividers. Verifies the rendering by re-reading the saved file. Refuses wall-of-text slides and skips-the-rendering-check. Trigger when the user asks for a .pptx, "make slides", "draft a deck for X", or to edit / convert an existing pptx.',
        author: 'sepilotd',
        tags: ['builtin', 'office', 'pptx', 'presentation', 'slides', 'python-pptx'],
        tools: [
          'fs.read',
          'fs.write',
          'fs.glob',
          'terminal.run',
          'office.list_open_documents',
          'office.read_active',
          'office.read_selection',
          'office.preview_edit',
          'office.apply_edit',
          'office.replace_selection',
          'computer.list_windows',
          'computer.focus_window',
          'computer.wait',
          'computer.observe',
          'memory.search',
          'memory.daily.append',
        ],
        autonomy_required: AutonomyLevel.Supervised,
        enabled: true,
      },
      content: `# PPTX Author
  
  Use this skill to generate or edit a Microsoft PowerPoint deck (\`.pptx\`). Backed by \`python-pptx\` for full structural fidelity (slide layouts, masters, notes, themes, tables, charts) — falls back to \`marp\` / \`reveal-md\` (Markdown → slides) when the user wants a lightweight code-first workflow.
  
  Do NOT use this skill to ship a 50-slide wall-of-text deck, to fabricate data points on chart slides, to overwrite an existing deck without confirming the user has a backup or version control, or to "convert PDF to pptx" (that's lossy and rarely good).
  
  ## Tool selection
  
  - **python-pptx** — default. Full fidelity over layouts, themes, masters, notes, tables, charts. Best when the deck must match a corporate template.
  - **Marp** (\`@marp-team/marp-cli\`) — Markdown → pptx / pdf / html. Best for code-driven decks (each slide is a heading + content block in Markdown). Less theme control.
  - **reveal-md** — same idea, more JS / web tooling. Use only if the user already lives there.
  
  Detect which is installed (\`terminal.run\` of \`python -c "import pptx" 2>&1\`, \`which marp\`, \`which reveal-md\`). If none, ask before installing into a local workspace virtual environment (for example \`python -m venv .sepilotd-office && .sepilotd-office/Scripts/python -m pip install python-pptx\` on Windows, or the equivalent POSIX path). Do not silently install globally.
  
  ## Operating principles
  
  - **1 idea per slide.** A slide with > 6 bullets is a document, not a slide.
  - **Speaker notes are first-class.** The slide carries the headline; notes carry what the speaker actually says. Both written together, neither as an afterthought.
  - **Reuse the corporate template when one exists.** \`python-pptx\` can open a \`.potx\` / existing \`.pptx\` as the base and inherit master / layouts / theme. Default to that when the user has a template file.
  - **Verify by re-reading.** After writing, re-open and walk the slides to confirm structure (slide count, layout names, bullet counts) matches the plan.
  - **Don't auto-generate charts from invented data.** If the user wants a bar chart, ask for the data — never fabricate plausible-looking numbers.
  
  ## Workflow
  
  1. **Establish the deck spec.** Audience, time budget (most decks ship 1 slide per 1–2 minutes), template (\`.potx\` / \`.pptx\` to inherit), title, brand tone.
  2. **Outline** as a numbered list: slide 1 = title, slide 2 = agenda, ... last slide = closing / call-to-action. Show outline; wait for approval before generating slides.
  3. **Generate** with python-pptx (or marp):
     - Use slide layouts named in the template (\`prs.slide_layouts[N]\`); do not hand-set positions when a layout exists.
     - Add speaker notes via \`slide.notes_slide.notes_text_frame.text\`.
     - For tables / charts: build from real data the user provided (or pulled from \`data-analyst\` skill output).
  4. **Save** to \`presentations/<title>-<date>.pptx\`. Save Markdown source too if Marp was used so future edits are diff-friendly.
  5. **Verify.** Re-open with python-pptx, walk slides + notes, confirm structure matches plan.
  6. **Report**: slide count, layouts used, total speaker-notes word count, file path, time-estimate.
  
  ## Editing an existing deck
  
  - \`fs.read\` the existing file is binary — use python-pptx to open it and walk it programmatically.
  - For each requested change, show the before / after summary (slide N: title was X, now Y; bullets added at slide N+1).
  - Per-slide approval for non-trivial changes.
  - Preserve the original (\`-original\` suffix) before in-place save.
  
  ## Editing an already-open PowerPoint deck
  
  Use the \`office.*\` live bridge only when the user is explicitly working with an open desktop Office app. Start with \`office.list_open_documents\`, focus the PowerPoint window with \`computer.list_windows\` / \`computer.focus_window\`, then call \`computer.wait\` and \`computer.observe(scope:"foreground")\` to understand the visible slide layout. Pair that visual evidence with \`office.read_active\` for exact deck text, or \`office.read_selection\` when the user has selected a shape/text region. The active deck must be saved inside the workspace by default; outside-workspace or unsaved decks require explicit user approval through the tool arguments and policy prompt. For any modification, call \`office.preview_edit\`, show the preview, then call \`office.apply_edit\` or \`office.replace_selection\` only with \`confirm=true\`. Re-observe after visual changes. Do not save the deck unless the user asked for saving (\`save=true\`).
  
  ## Anti-patterns to refuse
  
  - 7+ bullets on one slide.
  - Slides without speaker notes (it's a script, not a poster).
  - Fabricated chart data.
  - Hand-positioning text boxes when a layout would have worked.
  - "Make it pop" without the deck spec being clear first.
  - PDF → pptx conversion (lossy, brittle).
  - Overwriting an existing deck without backup confirmation.
  
  ## Pairs with
  
  - \`data-analyst\` — for any chart-bearing slide; produces the data + plot that this skill embeds.
  - \`technical-blog-writer\` — for the prose-to-slide adaptation (the same content reads very differently as slides vs as a blog post).
  - \`personal-writer\` / \`email-polish\` — for the cover email when distributing the deck.
  
  ## Final report
  
  - Tool used (python-pptx / Marp / reveal-md, with version).
  - Template inherited (path).
  - Slide count + layout breakdown.
  - Speaker-notes word count.
  - File saved (path).
  - Verification result (re-read structure matches plan).
  - Open data gaps surfaced to the user.
  `,
    },
  {
      metadata: {
        id: 'docx-author',
        name: 'docx-author',
        version: '1.3.0',
        description:
          'Author or edit Microsoft Word (.docx) documents via python-docx: headings, paragraphs, styled lists, tables, images, page breaks, headers / footers, table of contents, comments. Inherits an existing template (.dotx / .docx). Verifies the result by re-reading. Refuses hand-formatting when a style exists, refuses overwriting without backup. Trigger when the user asks for a .docx, "draft a Word doc", "make a report in Word format", or to edit an existing .docx.',
        author: 'sepilotd',
        tags: ['builtin', 'office', 'docx', 'word', 'report', 'python-docx'],
        tools: [
          'fs.read',
          'fs.write',
          'fs.glob',
          'terminal.run',
          'office.list_open_documents',
          'office.read_active',
          'office.read_selection',
          'office.preview_edit',
          'office.apply_edit',
          'office.replace_selection',
          'computer.list_windows',
          'computer.focus_window',
          'computer.wait',
          'computer.observe',
          'memory.search',
          'memory.daily.append',
        ],
        autonomy_required: AutonomyLevel.Supervised,
        enabled: true,
      },
      content: `# DOCX Author
  
  Use this skill to generate or edit a Microsoft Word document (\`.docx\`). Backed by \`python-docx\`. For one-shot Markdown-to-Word, \`pandoc -f markdown -t docx --reference-doc=<template>\` is the right tool — use it when the source is already Markdown.
  
  Do NOT use this skill to fabricate footnoted facts, to overwrite a doc without confirming a backup or version control, to hand-format text where a style ("Heading 1", "Caption", "Quote") would have applied, or to "convert PDF to docx" (lossy; suggest the user gets the source instead).
  
  ## Tool selection
  
  - **python-docx** — default. Programmatic control over styles, sections, tables, images, headers / footers, comments. Best when the doc must inherit a corporate template.
  - **pandoc** — when the input is Markdown / reST / LaTeX. \`pandoc -f markdown -t docx --reference-doc=template.docx\` preserves styling from the reference template. Fastest path for prose-heavy docs.
  
  Detect (\`python -c "import docx" 2>&1\`, \`which pandoc\`). If neither is available, ask before installing \`python-docx\` into a local workspace virtual environment (for example \`python -m venv .sepilotd-office && .sepilotd-office/Scripts/python -m pip install python-docx\` on Windows, or the equivalent POSIX path). Never install globally or silently.
  
  ## Operating principles
  
  - **Styles, not direct formatting.** \`paragraph.style = doc.styles['Heading 1']\` over hand-setting font / size / bold. The user's template's heading 1 becomes the heading 1.
  - **One source of truth.** The .docx is the deliverable; if the doc is regenerable from Markdown, keep both (commit the Markdown, regenerate the docx on demand). Mixing hand-edits into a regenerated doc is the start of merge pain.
  - **Track changes for collaborative edits.** When editing a doc someone else is using, preserve their tracked changes — never silently accept them all, never silently strip comments.
  - **Verify by re-reading.** Open the saved file, walk paragraphs + styles + table count, confirm against the plan.
  
  ## Workflow
  
  1. **Establish the doc spec.** Audience, length target, template (\`.dotx\` / reference \`.docx\`), structure (sections, headings).
  2. **Outline.** Heading 1 → Heading 2 → paragraphs. Show; approve; generate.
  3. **Generate.**
     - python-docx: \`doc.add_heading\`, \`doc.add_paragraph(style=...)\`, \`doc.add_table\`, \`doc.add_picture\`, header / footer via \`section.header\`. Use existing template styles.
     - pandoc: write Markdown to a temp file, \`pandoc input.md -o output.docx --reference-doc=template.docx\`.
  4. **Front matter.** Title page, TOC field (\`pandoc --toc\` or python-docx field code), revision date.
  5. **Save** to \`docs/<title>-<date>.docx\`. Keep Markdown source committed when applicable.
  6. **Verify.** Re-open and walk.
  
  ## Editing an existing doc
  
  - Open with python-docx; preserve the original (\`-original\` suffix).
  - Per-section approval for non-trivial edits.
  - For comments / tracked changes: read what's there, never silently accept-all-changes or delete-all-comments.
  
  ## Editing an already-open Word document
  
  Use the \`office.*\` live bridge only when the user explicitly asks to work with the open desktop Word document. Start with \`office.list_open_documents\`, focus the Word window with \`computer.list_windows\` / \`computer.focus_window\`, then call \`computer.wait\` and \`computer.observe(scope:"foreground")\` to understand the visible page, layout, cursor area, and comments pane. Pair that visual evidence with \`office.read_active\` for exact document text, or \`office.read_selection\` when the user has highlighted a paragraph/range. The active document must be saved inside the workspace by default; outside-workspace or unsaved documents require explicit user approval through the tool arguments and policy prompt. For any modification, call \`office.preview_edit\`, show the preview, then call \`office.apply_edit\` or \`office.replace_selection\` only with \`confirm=true\`. Re-observe after visual changes. Do not save the document unless the user asked for saving (\`save=true\`).
  
  ## Anti-patterns to refuse
  
  - Hand-set font/size/bold instead of using \`Heading 1\` / \`Caption\` style.
  - TOC entries that don't match the actual headings (regenerate the TOC from headings).
  - Pasted screenshots of tables instead of real Word tables (loses searchability + accessibility).
  - Overwriting without backup.
  - Silent accept-all-tracked-changes on a collaborator's document.
  - PDF → docx conversion (suggest getting the source).
  
  ## Pairs with
  
  - \`report-writer\` — drafts the prose; this skill renders it to .docx in the user's template.
  - \`doc-generator\` — for repo docs (README / CHANGELOG / API) — those stay Markdown; this skill is for shipped Word deliverables.
  - \`personal-writer\` — for letters / recommendations rendered as Word docs.
  - \`data-analyst\` — when the doc embeds a table or chart from data analysis.
  
  ## Final report
  
  - Tool used (python-docx / pandoc, with version).
  - Template inherited (path).
  - Section / heading count.
  - Page count (approximate).
  - File saved (path).
  - Verification (re-read structure matches plan).
  - Tracked changes / comments preserved? (yes / no — confirm).
  `,
    },
  {
      metadata: {
        id: 'xlsx-author',
        name: 'xlsx-author',
        version: '1.3.0',
        description:
          'Author or edit Microsoft Excel workbooks (.xlsx) via openpyxl / xlsxwriter / pandas. Multi-sheet, formulas (not hard-coded values), tables with header rows, data validation, conditional formatting, charts (bar / line / pie / scatter), named ranges, freeze panes, autofilter. Verifies by re-reading. Refuses hard-coded values where a formula would document the calculation. Trigger when the user asks for a .xlsx, "export to Excel", "make a spreadsheet for X", or to edit an existing workbook.',
        author: 'sepilotd',
        tags: ['builtin', 'office', 'xlsx', 'excel', 'spreadsheet', 'openpyxl'],
        tools: [
          'fs.read',
          'fs.write',
          'fs.glob',
          'terminal.run',
          'office.list_open_documents',
          'office.read_active',
          'office.read_selection',
          'office.preview_edit',
          'office.apply_edit',
          'office.replace_selection',
          'computer.list_windows',
          'computer.focus_window',
          'computer.wait',
          'computer.observe',
          'memory.search',
          'memory.daily.append',
        ],
        autonomy_required: AutonomyLevel.Supervised,
        enabled: true,
      },
      content: `# XLSX Author
  
  Use this skill to generate or edit a Microsoft Excel workbook (\`.xlsx\`). Backed by:
  
  - **openpyxl** — default. Full feature set (formulas, conditional formatting, data validation, charts, comments, named ranges, freeze panes). Best for structural workbooks.
  - **xlsxwriter** — when generating from scratch and you want richer charting + formatting (slightly nicer API). Cannot read existing files.
  - **pandas \`.to_excel()\`** — quickest path for "DataFrame → sheet" without bells and whistles. Built on openpyxl under the hood.
  
  Do NOT use this skill to hard-code computed values where a formula would document the calculation (\`=SUM(A2:A10)\` not \`42.7\`), to ship a workbook without a header row, to fabricate data, to overwrite an existing workbook without backup, or to use the macro-enabled \`.xlsm\` format unless the user explicitly asked for VBA.
  
  Detect dependencies before writing (\`python -c "import openpyxl"\`, \`python -c "import xlsxwriter"\`, \`python -c "import pandas"\`). If none are available, ask before installing \`openpyxl\` / \`xlsxwriter\` into a local workspace virtual environment (for example \`python -m venv .sepilotd-office && .sepilotd-office/Scripts/python -m pip install openpyxl xlsxwriter\` on Windows, or the equivalent POSIX path). Never install globally or silently.
  
  ## Operating principles
  
  - **Formulas, not values.** A cell that shows \`42.7\` should be \`=SUM(F2:F10)\` so the reader can audit. The exception: snapshot exports (clearly labelled "exported on YYYY-MM-DD").
  - **One concept per sheet.** "Data", "Calculations", "Summary", "Charts" — separate. A 20-column sheet with mixed inputs and outputs is unreadable.
  - **Header row + structured table.** Convert the data range to an Excel Table (\`ws.add_table\`) so filtering / sorting / formula references just work. Named ranges (\`workbook.defined_names\`) for anything referenced from another sheet.
  - **Data validation on input cells.** Drop-downs (list), number ranges, date ranges — catches typos at entry.
  - **Conditional formatting** to surface anomalies (red if < 0, yellow if missing).
  - **Freeze panes** on header rows so they stay visible while scrolling.
  
  ## Workflow
  
  1. **Establish the workbook spec.** Sheets + purpose of each, key calculations, charts requested, data sources, length / cell-count expectation.
  2. **Source the data.** From the user, from \`data-analyst\` output, from a CSV/JSON file (read with \`fs.read\` or \`pandas.read_csv\`).
  3. **Generate.**
     - Create the workbook (\`openpyxl.Workbook()\`).
     - For each sheet: header row first (bold + frozen), then data, then totals row with formulas, then named ranges.
     - Charts last (they reference cell ranges that must already exist).
     - Apply column widths (auto-fit or user-specified), zoom level, sheet view.
  4. **Save** to \`spreadsheets/<title>-<date>.xlsx\`.
  5. **Verify.** Re-open with openpyxl, check formulas evaluated correctly (load with \`data_only=True\` to read computed values; without it, you read the formula string).
  
  ## Editing an existing workbook
  
  - \`load_workbook(path)\` — preserves existing structure including formulas, conditional formats, charts.
  - For complex edits: write a backup \`-original.xlsx\` first.
  - Per-sheet approval for non-trivial changes.
  - Be careful with formula references when inserting / deleting rows — openpyxl does NOT auto-update references the way Excel UI does. Insert / delete operations may need explicit formula adjustment.
  
  ## Editing an already-open Excel workbook
  
  Use the \`office.*\` live bridge only when the user explicitly asks to work with the open desktop Excel workbook. Start with \`office.list_open_documents\`, focus the Excel window with \`computer.list_windows\` / \`computer.focus_window\`, then call \`computer.wait\` and \`computer.observe(scope:"foreground")\` to understand the visible grid, charts, frozen panes, filters, and selected cells. Pair that visual evidence with \`office.read_active\` for exact active-sheet values, or \`office.read_selection\` when the user has selected a cell/range. The active workbook must be saved inside the workspace by default; outside-workspace or unsaved workbooks require explicit user approval through the tool arguments and policy prompt. For any modification, call \`office.preview_edit\`, show the preview, then call \`office.apply_edit\` or \`office.replace_selection\` only with \`confirm=true\`. Prefer formulas over pasted values, re-observe after visual changes, and do not save the workbook unless the user asked for saving (\`save=true\`).
  
  ## Anti-patterns to refuse
  
  - Hard-coding the sum / average / count when a formula would document it.
  - A 30-column "everything" sheet — split it.
  - No header row (or merged-cell pseudo-headers — breaks filtering).
  - Data validation skipped on free-form input columns.
  - Overwriting an existing workbook without backup.
  - Fabricated data to "fill out" a sheet.
  - Charts referencing cells that don't exist yet.
  - Macro-enabled .xlsm without the user explicitly requesting VBA.
  
  ## Pairs with
  
  - \`data-analyst\` — produces the dataset; this skill renders it into a workbook with formulas, charts, validation. Many "give me Excel" requests are really data-analyst questions where Excel is just the delivery format.
  - \`report-writer\` — when the deliverable is "report + supporting spreadsheet", the report references this skill's output by sheet:cell reference.
  - \`pptx-author\` — when a chart needs to live in both a deck and a workbook, build it in the workbook first and reference it from the deck.
  
  ## Final report
  
  - Tool used (openpyxl / xlsxwriter / pandas, with versions).
  - Sheets + their role.
  - Formula count vs hard-coded-value count (the more formulas, the more auditable).
  - Charts / named ranges / data-validations / conditional-formats added (counts).
  - File saved (path).
  - Verification (load with data_only=True; sample-check 3 computed cells against expected).
  - Any "should be a formula but I had no data" cells flagged.
  `,
    },
]

export async function seedBuiltinSkills(
  registry: FileSkillRegistry,
  skills: BuiltinSkill[] = builtinSkills,
): Promise<void> {
  if (skills === builtinSkills) {
    const shippedIds = new Set(skills.map(skill => skill.metadata.id))
    for (const existing of await registry.listAll()) {
      if (existing.author === 'sepilotd' && !shippedIds.has(existing.id)) {
        await registry.remove(existing.id)
      }
    }
  }
  for (const builtin of skills) {
    const existing = await registry.get(builtin.metadata.id)
    if (existing && !shouldUpdateManagedBuiltin(existing, builtin)) {
      continue
    }
    // Preserve a user-toggled enabled flag across re-seeds; fall back to the
    // built-in's declared default only on first install.
    const enabled = existing?.metadata.enabled ?? builtin.metadata.enabled
    await registry.register({ ...builtin.metadata, enabled }, builtin.content, { force: true })
  }
}

function shouldUpdateManagedBuiltin(
  existing: BuiltinSkill,
  builtin: BuiltinSkill,
): boolean {
  if (existing.metadata.author !== 'sepilotd') return false
  const versionComparison = compareSemver(
    existing.metadata.version,
    builtin.metadata.version,
  )
  if (versionComparison < 0) return true
  if (versionComparison > 0) return false

  // Runtime policy trusts a managed built-in only when its shipped metadata
  // and prompt body match the canonical bundle. Keeping same-version drift on
  // disk would make seeding call it current while policy silently rejects it.
  // Reconcile that split-brain state, but preserve the user's enabled toggle
  // when register() writes the canonical definition below.
  const { enabled: _existingEnabled, ...existingIdentity } = existing.metadata
  const { enabled: _builtinEnabled, ...builtinIdentity } = builtin.metadata
  return (
    !isDeepStrictEqual(existingIdentity, builtinIdentity)
    || normalizeManagedBuiltinContent(existing.content)
      !== normalizeManagedBuiltinContent(builtin.content)
  )
}

function normalizeManagedBuiltinContent(content: string): string {
  return content.replace(/\r\n/g, '\n').trim()
}

function compareSemver(a: string, b: string): number {
  const left = parseSemver(a)
  const right = parseSemver(b)
  for (let i = 0; i < 3; i += 1) {
    if (left[i] !== right[i]) return left[i] - right[i]
  }
  return 0
}

function parseSemver(value: string): [number, number, number] {
  const [major = '0', minor = '0', patch = '0'] = value.split('.')
  return [
    Number.parseInt(major, 10) || 0,
    Number.parseInt(minor, 10) || 0,
    Number.parseInt(patch, 10) || 0,
  ]
}
