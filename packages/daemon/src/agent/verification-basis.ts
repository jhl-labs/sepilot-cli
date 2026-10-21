const CONTRACT_EVIDENCE = '[Independent contract evidence] New files also have external compatibility obligations. Establish public conventions from the explicit user contract, existing consumers or an installed dependency before choosing a representation.'

const EXPECTATION_ORIGIN = 'Name the independently established source of expected results. Implementation-authored comments, tests and the planner\'s inferred contract are not independent authority for details absent from the user request. Preserve explicit user overrides; otherwise check the relevant library/API reference rather than extrapolating a convention.'

const PROPORTIONAL_CHECKS = 'Do not invent an external standard or require network access when the explicit contract is sufficient. A read-back or diff can suffice for literal text or documentation changes.'

export const IMPLEMENTATION_CONTRACT_GUIDANCE = [
  CONTRACT_EVIDENCE,
  'Preserve explicit user overrides. Check material uncertainties against the relevant reference, not your own comments or the planner\'s assumptions; reuse sufficient evidence.',
].join(' ')

export const VALIDATION_REFERENCE_GUIDANCE = [
  CONTRACT_EVIDENCE,
  EXPECTATION_ORIGIN,
  'A test oracle copied from the implementation proves self-consistency, not an external convention. Choose inputs that distinguish plausible competing interpretations and compare to the reference or a derivation from an explicit specification.',
  PROPORTIONAL_CHECKS,
].join(' ')

export const QUALITY_REFERENCE_GUIDANCE = [
  VALIDATION_REFERENCE_GUIDANCE,
  'No existing callers does not exempt a new public interface. Assess compatibility with its required external contract, not just preservation of previous behavior. Missing reference evidence for a material convention means changeImpactStatus=unverified and nextPhase=validation; an observed contradiction means incompatible and nextPhase=implementation. Reserve not_applicable for evidenced non-behavioral changes, not new implementations. Never promote self-consistency into reference compatibility.',
].join(' ')
