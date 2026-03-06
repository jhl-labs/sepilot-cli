"""Debate Pattern - Multi-Perspective Code Review and Problem Solving

This module implements the debate pattern for improved decision making:
- Proposer: Generates initial solution
- Critic: Analyzes and critiques the proposal
- Resolver: Synthesizes feedback into final decision

Ideal for:
- Code review quality improvement
- Bug fix verification
- Architecture decisions
- Complex problem solving

Inspired by:
- Debate: AI Safety via Debate
- Multi-Agent Debate for Code Generation
- Self-Consistency and Multiple Perspectives
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from sepilot.agent.enhanced_state import EnhancedAgentState


def _extract_json_block(content: str) -> str:
    """Safely extract JSON block from LLM response content.

    Handles ```json ... ``` and bare ``` ... ``` blocks.
    Falls back to raw content if no code block found.
    """
    import re
    # Try ```json ... ``` first, then ``` ... ```
    match = re.search(r'```json\s*(.*?)```', content, re.DOTALL)
    if not match:
        match = re.search(r'```\s*(.*?)```', content, re.DOTALL)
    return match.group(1).strip() if match else content


class DebateRole(str, Enum):
    """Roles in the debate."""
    PROPOSER = "proposer"
    CRITIC = "critic"
    RESOLVER = "resolver"


class DebateOutcome(str, Enum):
    """Possible debate outcomes."""
    APPROVED = "approved"           # Proposal accepted
    REJECTED = "rejected"           # Proposal rejected
    REVISED = "revised"             # Proposal needs revision
    NEEDS_MORE_INFO = "needs_more_info"  # Need more information
    ESCALATE = "escalate"           # Human decision needed


@dataclass
class DebateArgument:
    """A single argument in the debate."""
    role: DebateRole
    content: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DebateRound:
    """A complete round of debate."""
    round_number: int
    proposal: DebateArgument
    critique: DebateArgument
    resolution: DebateArgument | None = None
    outcome: DebateOutcome | None = None


@dataclass
class DebateResult:
    """Final result of the debate process."""
    topic: str
    rounds: list[DebateRound]
    final_outcome: DebateOutcome
    final_decision: str
    confidence: float
    key_points: list[str]
    dissenting_views: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "total_rounds": len(self.rounds),
            "final_outcome": self.final_outcome.value,
            "final_decision": self.final_decision,
            "confidence": self.confidence,
            "key_points": self.key_points,
            "dissenting_views": self.dissenting_views,
            "recommendations": self.recommendations
        }


class ProposerAgent:
    """Agent that proposes solutions."""

    SYSTEM_PROMPT = """You are the PROPOSER in a code review debate.

Your role:
- Analyze the code/task and propose a solution
- Be specific and actionable
- Provide clear reasoning for your proposal
- Anticipate potential concerns

Output format (JSON):
{
    "proposal": "Your proposed solution or action",
    "reasoning": "Why this is the best approach",
    "benefits": ["benefit1", "benefit2"],
    "potential_risks": ["risk1", "risk2"],
    "confidence": 0.0-1.0
}"""

    def __init__(self, llm: BaseChatModel):
        """Initialize proposer agent."""
        self.llm = llm

    def propose(self, context: str, previous_feedback: str | None = None) -> DebateArgument:
        """Generate a proposal.

        Args:
            context: The context/code to analyze
            previous_feedback: Feedback from previous round (if any)

        Returns:
            DebateArgument with proposal
        """
        user_prompt = f"Context:\n{context}"
        if previous_feedback:
            user_prompt += f"\n\nPrevious critique to address:\n{previous_feedback}"

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            import json
            content = _extract_json_block(response.content.strip())

            data = json.loads(content)

            return DebateArgument(
                role=DebateRole.PROPOSER,
                content=data.get("proposal", ""),
                confidence=data.get("confidence", 0.7),
                supporting_evidence=data.get("benefits", []),
                concerns=data.get("potential_risks", [])
            )

        except Exception as e:
            return DebateArgument(
                role=DebateRole.PROPOSER,
                content=f"Error generating proposal: {e}",
                confidence=0.3
            )


class CriticAgent:
    """Agent that critiques proposals."""

    SYSTEM_PROMPT = """You are the CRITIC in a code review debate.

Your role:
- Carefully analyze the proposal
- Identify weaknesses, risks, and edge cases
- Suggest improvements
- Be constructive but thorough

Output format (JSON):
{
    "assessment": "overall|good|concerning|problematic",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "security_concerns": ["concern1"],
    "performance_concerns": ["concern1"],
    "maintainability_issues": ["issue1"],
    "suggested_improvements": ["improvement1", "improvement2"],
    "verdict": "approve|reject|revise",
    "confidence": 0.0-1.0
}"""

    def __init__(self, llm: BaseChatModel):
        """Initialize critic agent."""
        self.llm = llm

    def critique(self, proposal: DebateArgument, context: str) -> DebateArgument:
        """Critique a proposal.

        Args:
            proposal: The proposal to critique
            context: Original context

        Returns:
            DebateArgument with critique
        """
        user_prompt = f"""Context:
{context}

Proposal to critique:
{proposal.content}

Proposer's confidence: {proposal.confidence:.0%}
Stated benefits: {', '.join(proposal.supporting_evidence)}
Stated risks: {', '.join(proposal.concerns)}

Provide your critical analysis."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            import json
            content = _extract_json_block(response.content.strip())

            data = json.loads(content)

            concerns = []
            concerns.extend(data.get("weaknesses", []))
            concerns.extend(data.get("security_concerns", []))
            concerns.extend(data.get("performance_concerns", []))
            concerns.extend(data.get("maintainability_issues", []))

            return DebateArgument(
                role=DebateRole.CRITIC,
                content=f"Assessment: {data.get('assessment', 'unknown')}. Verdict: {data.get('verdict', 'revise')}",
                confidence=data.get("confidence", 0.7),
                supporting_evidence=data.get("strengths", []),
                concerns=concerns[:10]  # Limit concerns
            )

        except Exception as e:
            return DebateArgument(
                role=DebateRole.CRITIC,
                content=f"Error generating critique: {e}",
                confidence=0.3
            )


class ResolverAgent:
    """Agent that synthesizes debate and makes final decision."""

    SYSTEM_PROMPT = """You are the RESOLVER in a code review debate.

Your role:
- Consider both the proposal and critique
- Weigh the arguments objectively
- Make a final decision
- Provide clear reasoning

Output format (JSON):
{
    "decision": "approve|reject|revise|escalate",
    "reasoning": "Why this decision was made",
    "key_considerations": ["point1", "point2"],
    "action_items": ["action1", "action2"],
    "dissenting_points": ["point that wasn't fully addressed"],
    "confidence": 0.0-1.0,
    "final_recommendation": "Specific recommendation for implementation"
}"""

    def __init__(self, llm: BaseChatModel):
        """Initialize resolver agent."""
        self.llm = llm

    def resolve(
        self,
        proposal: DebateArgument,
        critique: DebateArgument,
        context: str
    ) -> tuple[DebateArgument, DebateOutcome]:
        """Resolve the debate.

        Args:
            proposal: The proposal
            critique: The critique
            context: Original context

        Returns:
            Tuple of (resolution argument, outcome)
        """
        user_prompt = f"""Context:
{context}

PROPOSAL:
{proposal.content}
Confidence: {proposal.confidence:.0%}
Benefits: {', '.join(proposal.supporting_evidence)}
Risks: {', '.join(proposal.concerns)}

CRITIQUE:
{critique.content}
Confidence: {critique.confidence:.0%}
Strengths found: {', '.join(critique.supporting_evidence)}
Concerns raised: {', '.join(critique.concerns)}

Make your final decision."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            import json
            content = _extract_json_block(response.content.strip())

            data = json.loads(content)

            decision = data.get("decision", "revise")
            outcome_map = {
                "approve": DebateOutcome.APPROVED,
                "reject": DebateOutcome.REJECTED,
                "revise": DebateOutcome.REVISED,
                "escalate": DebateOutcome.ESCALATE
            }
            outcome = outcome_map.get(decision, DebateOutcome.REVISED)

            argument = DebateArgument(
                role=DebateRole.RESOLVER,
                content=data.get("final_recommendation", ""),
                confidence=data.get("confidence", 0.7),
                supporting_evidence=data.get("key_considerations", []),
                concerns=data.get("dissenting_points", [])
            )

            return argument, outcome

        except Exception as e:
            return DebateArgument(
                role=DebateRole.RESOLVER,
                content=f"Error resolving debate: {e}",
                confidence=0.3
            ), DebateOutcome.ESCALATE


class DebateOrchestrator:
    """Orchestrates the debate process.

    Coordinates proposer, critic, and resolver to reach
    a well-reasoned decision through structured debate.
    """

    MAX_ROUNDS = 2  # Maximum debate rounds (reduced from 3 for efficiency)

    def __init__(
        self,
        llm: BaseChatModel,
        console: Any | None = None,
        verbose: bool = False
    ):
        """Initialize debate orchestrator.

        Args:
            llm: Language model for all agents
            console: Rich console for output
            verbose: Verbose output flag
        """
        self.proposer = ProposerAgent(llm)
        self.critic = CriticAgent(llm)
        self.resolver = ResolverAgent(llm)
        self.console = console
        self.verbose = verbose

    def conduct_debate(
        self,
        topic: str,
        context: str,
        min_confidence: float = 0.7
    ) -> DebateResult:
        """Conduct a full debate on a topic.

        Args:
            topic: Topic/question for debate
            context: Relevant context (code, requirements)
            min_confidence: Minimum confidence for acceptance

        Returns:
            DebateResult with final decision
        """
        rounds: list[DebateRound] = []
        previous_feedback = None

        for round_num in range(1, self.MAX_ROUNDS + 1):
            if self.console and self.verbose:
                self.console.print(f"[cyan]🎭 Debate Round {round_num}...[/cyan]")

            # Step 1: Proposer generates/revises proposal
            proposal = self.proposer.propose(context, previous_feedback)

            if self.console and self.verbose:
                self.console.print(
                    f"  [green]📝 Proposer:[/green] {proposal.content[:100]}... "
                    f"(confidence: {proposal.confidence:.0%})"
                )

            # Step 2: Critic analyzes proposal
            critique = self.critic.critique(proposal, context)

            if self.console and self.verbose:
                self.console.print(
                    f"  [yellow]🔍 Critic:[/yellow] {critique.content[:100]}... "
                    f"(concerns: {len(critique.concerns)})"
                )

            # Step 3: Resolver makes decision
            resolution, outcome = self.resolver.resolve(proposal, critique, context)

            if self.console and self.verbose:
                self.console.print(
                    f"  [blue]⚖️ Resolver:[/blue] {outcome.value} "
                    f"(confidence: {resolution.confidence:.0%})"
                )

            round_result = DebateRound(
                round_number=round_num,
                proposal=proposal,
                critique=critique,
                resolution=resolution,
                outcome=outcome
            )
            rounds.append(round_result)

            # Check if we've reached a decision — break immediately on any decision
            if outcome in [DebateOutcome.APPROVED, DebateOutcome.REJECTED]:
                break  # Decision reached, no need for confidence threshold
            elif outcome == DebateOutcome.ESCALATE:
                break

            # Prepare feedback for next round
            previous_feedback = f"""
Previous outcome: {outcome.value}
Concerns to address: {', '.join(critique.concerns[:5])}
Resolver feedback: {resolution.content}
"""

        # Compile final result
        final_round = rounds[-1]
        final_outcome = final_round.outcome or DebateOutcome.REVISED
        final_resolution = final_round.resolution

        return DebateResult(
            topic=topic,
            rounds=rounds,
            final_outcome=final_outcome,
            final_decision=final_resolution.content if final_resolution else "No decision reached",
            confidence=final_resolution.confidence if final_resolution else 0.5,
            key_points=self._extract_key_points(rounds),
            dissenting_views=self._extract_dissenting_views(rounds),
            recommendations=self._extract_recommendations(rounds)
        )

    def _extract_key_points(self, rounds: list[DebateRound]) -> list[str]:
        """Extract key points from debate."""
        points = []
        for round_item in rounds:
            points.extend(round_item.proposal.supporting_evidence[:2])
            if round_item.resolution:
                points.extend(round_item.resolution.supporting_evidence[:2])
        return list(set(points))[:5]

    def _extract_dissenting_views(self, rounds: list[DebateRound]) -> list[str]:
        """Extract dissenting views from debate."""
        views = []
        for round_item in rounds:
            views.extend(round_item.critique.concerns[:2])
        return list(set(views))[:3]

    def _extract_recommendations(self, rounds: list[DebateRound]) -> list[str]:
        """Extract recommendations from debate."""
        recs = []
        final_round = rounds[-1]
        if final_round.resolution:
            recs.append(final_round.resolution.content)
        return recs


class DebateNode:
    """LangGraph node for debate-based decision making."""

    def __init__(
        self,
        orchestrator: DebateOrchestrator,
        trigger_keywords: list[str] | None = None,
        console: Any | None = None,
        verbose: bool = False
    ):
        """Initialize debate node.

        Args:
            orchestrator: DebateOrchestrator instance
            trigger_keywords: Keywords that trigger debate mode
            console: Rich console
            verbose: Verbose output flag
        """
        self.orchestrator = orchestrator
        self.trigger_keywords = trigger_keywords or [
            "review", "check", "verify", "analyze",
            "security", "performance", "architecture"
        ]
        self.console = console
        self.verbose = verbose

    def should_debate(self, state: EnhancedAgentState) -> bool:
        """Determine if debate should be triggered.

        Args:
            state: Current agent state

        Returns:
            True if debate should be conducted
        """
        # Get task description
        messages = state.get("messages", [])
        task = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task = getattr(msg, "content", "").lower()
                break

        # Check for trigger keywords
        return any(kw in task for kw in self.trigger_keywords)

    def __call__(self, state: EnhancedAgentState) -> dict[str, Any]:
        """Execute debate if appropriate.

        Args:
            state: Current agent state

        Returns:
            State updates with debate result
        """
        if not self.should_debate(state):
            return {}

        # Get context
        messages = state.get("messages", [])
        task = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                task = getattr(msg, "content", "")
                break

        # Get relevant code context
        file_changes = state.get("file_changes", [])
        code_context = ""
        for fc in file_changes[:3]:
            if fc.new_content:
                code_context += f"\n--- {fc.file_path} ---\n{fc.new_content[:1000]}\n"

        # Conduct debate
        context = f"Task: {task}\n\nCode:\n{code_context}"
        result = self.orchestrator.conduct_debate(
            topic=task[:200],
            context=context
        )

        if self.console and self.verbose:
            self.console.print(
                f"[bold cyan]🎭 Debate complete: {result.final_outcome.value} "
                f"(confidence: {result.confidence:.0%})[/bold cyan]"
            )

        return {
            "debate_result": result.to_dict(),
            "debate_decision": result.final_decision,
            "debate_confidence": result.confidence,
            "planning_notes": [
                f"Debate outcome: {result.final_outcome.value}",
                f"Key points: {'; '.join(result.key_points[:3])}"
            ]
        }


def create_debate_orchestrator(
    llm: BaseChatModel,
    console: Any | None = None,
    verbose: bool = False
) -> DebateOrchestrator:
    """Factory function to create DebateOrchestrator.

    Args:
        llm: Language model
        console: Rich console
        verbose: Verbose output

    Returns:
        Configured DebateOrchestrator
    """
    return DebateOrchestrator(llm=llm, console=console, verbose=verbose)
