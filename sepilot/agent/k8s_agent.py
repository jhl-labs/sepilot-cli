"""Kubernetes Agent - AI-powered Kubernetes cluster health monitoring

This module provides intelligent Kubernetes cluster health checks with:
- Comprehensive cluster health analysis
- Pod, Node, and Service status monitoring
- Event analysis for troubleshooting
- AI-powered issue diagnosis and recommendations
"""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sepilot.config.settings import Settings
from sepilot.loggers.file_logger import FileLogger


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ResourceHealth:
    """Health status of a Kubernetes resource."""

    name: str
    namespace: str
    kind: str
    status: HealthStatus
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ClusterHealth:
    """Overall cluster health report."""

    status: HealthStatus
    nodes: list[ResourceHealth] = field(default_factory=list)
    pods: list[ResourceHealth] = field(default_factory=list)
    services: list[ResourceHealth] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class K8sAgent:
    """AI-powered Kubernetes cluster health monitoring agent.

    Features:
    - Cluster-wide health assessment
    - Node status monitoring
    - Pod health and restart analysis
    - Service endpoint verification
    - Event log analysis
    - AI-powered diagnosis and recommendations
    """

    def __init__(
        self,
        settings: Settings,
        logger: FileLogger,
        console: Console | None = None,
        kubeconfig: str | Path | None = None,
        context: str | None = None,
    ):
        """Initialize K8sAgent.

        Args:
            settings: Application settings
            logger: File logger instance
            console: Rich console for output
            kubeconfig: Path to kubeconfig file
            context: Kubernetes context to use
        """
        self.settings = settings
        self.logger = logger
        self.console = console or Console()
        self.kubeconfig = kubeconfig
        self.context = context

        # LLM for AI analysis
        self._llm = None

    def _get_llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            from sepilot.llm.factory import create_llm

            self._llm = create_llm(
                model=self.settings.model,
                temperature=0.3,
            )
        return self._llm

    def _run_kubectl(self, *args: str, capture: bool = True) -> tuple[bool, str]:
        """Run a kubectl command.

        Args:
            *args: kubectl command arguments
            capture: Whether to capture output

        Returns:
            Tuple of (success, output)
        """
        cmd = ["kubectl"]

        if self.kubeconfig:
            cmd.extend(["--kubeconfig", str(self.kubeconfig)])
        if self.context:
            cmd.extend(["--context", self.context])

        cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture,
                text=True,
                timeout=60,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, "kubectl not found. Please install kubectl."
        except Exception as e:
            return False, str(e)

    def _check_cluster_connection(self) -> bool:
        """Check if we can connect to the cluster.

        Returns:
            True if connected
        """
        success, output = self._run_kubectl("cluster-info", "--request-timeout=5s")
        return success

    def _get_current_context(self) -> str:
        """Get current Kubernetes context.

        Returns:
            Context name or empty string
        """
        success, output = self._run_kubectl("config", "current-context")
        if success:
            return output.strip()
        return ""

    def _get_nodes_health(self) -> list[ResourceHealth]:
        """Get health status of all nodes.

        Returns:
            List of ResourceHealth for nodes
        """
        nodes = []

        success, output = self._run_kubectl(
            "get", "nodes", "-o",
            "jsonpath={range .items[*]}{.metadata.name}|{.status.conditions[-1].type}|"
            "{.status.conditions[-1].status}|{.status.conditions[-1].message}\\n{end}"
        )

        if not success:
            return nodes

        for line in output.strip().split("\n"):
            if not line or "|" not in line:
                continue

            parts = line.split("|")
            if len(parts) >= 4:
                name, condition_type, condition_status, message = parts[:4]

                if condition_type == "Ready" and condition_status == "True":
                    status = HealthStatus.HEALTHY
                elif condition_type == "Ready" and condition_status == "False":
                    status = HealthStatus.CRITICAL
                else:
                    status = HealthStatus.WARNING

                nodes.append(ResourceHealth(
                    name=name,
                    namespace="",
                    kind="Node",
                    status=status,
                    message=message[:200] if message else f"Condition: {condition_type}={condition_status}",
                ))

        return nodes

    def _get_pods_health(self, namespace: str = "") -> list[ResourceHealth]:
        """Get health status of pods.

        Args:
            namespace: Namespace to check (empty for all)

        Returns:
            List of ResourceHealth for unhealthy pods
        """
        pods = []

        ns_args = ["-n", namespace] if namespace else ["-A"]

        success, output = self._run_kubectl(
            "get", "pods", *ns_args, "-o",
            "jsonpath={range .items[*]}{.metadata.namespace}|{.metadata.name}|"
            "{.status.phase}|{.status.containerStatuses[0].restartCount}|"
            "{.status.containerStatuses[0].state}\\n{end}"
        )

        if not success:
            return pods

        for line in output.strip().split("\n"):
            if not line or "|" not in line:
                continue

            parts = line.split("|")
            if len(parts) >= 4:
                ns, name, phase, restart_count = parts[:4]

                try:
                    restarts = int(restart_count) if restart_count else 0
                except ValueError:
                    restarts = 0

                # Determine health status
                if phase == "Running" and restarts < 5:
                    status = HealthStatus.HEALTHY
                    message = f"Running (restarts: {restarts})"
                elif phase == "Running" and restarts >= 5:
                    status = HealthStatus.WARNING
                    message = f"Running but high restart count: {restarts}"
                elif phase in ["Pending", "Unknown"]:
                    status = HealthStatus.WARNING
                    message = f"Phase: {phase}"
                elif phase in ["Failed", "CrashLoopBackOff"]:
                    status = HealthStatus.CRITICAL
                    message = f"Phase: {phase}"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Phase: {phase}"

                # Only include non-healthy pods
                if status != HealthStatus.HEALTHY:
                    pods.append(ResourceHealth(
                        name=name,
                        namespace=ns,
                        kind="Pod",
                        status=status,
                        message=message,
                        details={"restarts": restarts, "phase": phase},
                    ))

        return pods

    def _get_services_health(self, namespace: str = "") -> list[ResourceHealth]:
        """Get health status of services.

        Args:
            namespace: Namespace to check (empty for all)

        Returns:
            List of ResourceHealth for services with issues
        """
        services = []

        ns_args = ["-n", namespace] if namespace else ["-A"]

        # Get services without endpoints
        success, output = self._run_kubectl(
            "get", "endpoints", *ns_args, "-o",
            "jsonpath={range .items[*]}{.metadata.namespace}|{.metadata.name}|"
            "{.subsets[0].addresses}\\n{end}"
        )

        if not success:
            return services

        for line in output.strip().split("\n"):
            if not line or "|" not in line:
                continue

            parts = line.split("|")
            if len(parts) >= 3:
                ns, name, addresses = parts[:3]

                # Skip kubernetes system service
                if name == "kubernetes" and ns == "default":
                    continue

                if not addresses or addresses == "<none>" or addresses == "":
                    services.append(ResourceHealth(
                        name=name,
                        namespace=ns,
                        kind="Service",
                        status=HealthStatus.WARNING,
                        message="No endpoints available",
                    ))

        return services

    def _get_recent_events(self, namespace: str = "", limit: int = 20) -> list[dict]:
        """Get recent warning/error events.

        Args:
            namespace: Namespace to check (empty for all)
            limit: Maximum events to return

        Returns:
            List of event dicts
        """
        events = []

        ns_args = ["-n", namespace] if namespace else ["-A"]

        success, output = self._run_kubectl(
            "get", "events", *ns_args,
            "--field-selector=type!=Normal",
            "--sort-by=.lastTimestamp",
            "-o", "jsonpath={range .items[*]}{.metadata.namespace}|{.involvedObject.kind}|"
            "{.involvedObject.name}|{.type}|{.reason}|{.message}\\n{end}"
        )

        if not success:
            return events

        for line in output.strip().split("\n")[:limit]:
            if not line or "|" not in line:
                continue

            parts = line.split("|")
            if len(parts) >= 6:
                ns, kind, name, event_type, reason, message = parts[:6]
                events.append({
                    "namespace": ns,
                    "kind": kind,
                    "name": name,
                    "type": event_type,
                    "reason": reason,
                    "message": message[:200],
                })

        return events

    def _analyze_with_ai(self, health: ClusterHealth) -> tuple[str, list[str]]:
        """Use AI to analyze cluster health and provide recommendations.

        Args:
            health: ClusterHealth data

        Returns:
            Tuple of (summary, recommendations)
        """
        llm = self._get_llm()

        # Build context
        context_parts = []

        # Nodes
        if health.nodes:
            node_summary = "\n".join([
                f"- {n.name}: {n.status.value} - {n.message}"
                for n in health.nodes
            ])
            context_parts.append(f"## Nodes\n{node_summary}")

        # Unhealthy Pods
        if health.pods:
            pod_summary = "\n".join([
                f"- {p.namespace}/{p.name}: {p.status.value} - {p.message}"
                for p in health.pods[:20]  # Limit for token efficiency
            ])
            context_parts.append(f"## Unhealthy Pods\n{pod_summary}")

        # Services without endpoints
        if health.services:
            svc_summary = "\n".join([
                f"- {s.namespace}/{s.name}: {s.message}"
                for s in health.services[:10]
            ])
            context_parts.append(f"## Services Issues\n{svc_summary}")

        # Events
        if health.events:
            event_summary = "\n".join([
                f"- [{e['type']}] {e['namespace']}/{e['kind']}/{e['name']}: {e['reason']} - {e['message']}"
                for e in health.events[:10]
            ])
            context_parts.append(f"## Recent Warning/Error Events\n{event_summary}")

        if not context_parts:
            return "Cluster appears healthy with no significant issues detected.", []

        context = "\n\n".join(context_parts)

        system_prompt = """You are a senior Kubernetes administrator helping to diagnose cluster issues.
Analyze the provided cluster health data and provide:
1. A brief summary (2-3 sentences) of the overall cluster health
2. A prioritized list of recommendations to fix issues

Be concise and actionable. Focus on the most critical issues first.
Use Korean for your response."""

        user_prompt = f"""Kubernetes 클러스터 상태를 분석해주세요:

{context}

다음 형식으로 응답해주세요:

## 요약
(2-3문장 요약)

## 권장 조치
1. (가장 중요한 조치)
2. (다음 조치)
..."""

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            # Parse response
            content = response.content
            summary = ""
            recommendations = []

            if "## 요약" in content:
                parts = content.split("## 권장 조치")
                summary_part = parts[0].replace("## 요약", "").strip()
                summary = summary_part

                if len(parts) > 1:
                    rec_lines = parts[1].strip().split("\n")
                    for line in rec_lines:
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith("-")):
                            # Remove numbering
                            rec = line.lstrip("0123456789.-) ").strip()
                            if rec:
                                recommendations.append(rec)

            return summary or content[:500], recommendations[:10]

        except Exception as e:
            return f"AI 분석 실패: {e}", []

    def run_health_check(
        self,
        namespace: str = "",
        include_ai: bool = True,
        verbose: bool = False,
    ) -> ClusterHealth:
        """Run comprehensive cluster health check.

        Args:
            namespace: Specific namespace to check (empty for cluster-wide)
            include_ai: Include AI analysis
            verbose: Show detailed output

        Returns:
            ClusterHealth report
        """
        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]Kubernetes Cluster Health Check[/bold cyan]\n\n"
                "Analyzing cluster resources for potential issues...",
                title="/k8s-health",
                border_style="cyan",
            )
        )

        # Check connection
        self.console.print("\n[bold]1. Checking cluster connection...[/bold]")
        if not self._check_cluster_connection():
            self.console.print("[red]Failed to connect to Kubernetes cluster[/red]")
            self.console.print("[dim]Check your kubeconfig and cluster availability[/dim]")
            return ClusterHealth(
                status=HealthStatus.UNKNOWN,
                summary="Cannot connect to cluster",
            )

        context = self._get_current_context()
        self.console.print(f"   Connected to: [cyan]{context}[/cyan]")

        # Collect health data
        self.console.print("\n[bold]2. Checking node health...[/bold]")
        nodes = self._get_nodes_health()
        healthy_nodes = sum(1 for n in nodes if n.status == HealthStatus.HEALTHY)
        self.console.print(f"   Nodes: {healthy_nodes}/{len(nodes)} healthy")

        self.console.print("\n[bold]3. Checking pod health...[/bold]")
        pods = self._get_pods_health(namespace)
        self.console.print(f"   Found {len(pods)} unhealthy pods")

        self.console.print("\n[bold]4. Checking service endpoints...[/bold]")
        services = self._get_services_health(namespace)
        self.console.print(f"   Found {len(services)} services without endpoints")

        self.console.print("\n[bold]5. Checking recent events...[/bold]")
        events = self._get_recent_events(namespace)
        self.console.print(f"   Found {len(events)} warning/error events")

        # Determine overall status
        if any(n.status == HealthStatus.CRITICAL for n in nodes):
            overall_status = HealthStatus.CRITICAL
        elif pods or services or any(n.status == HealthStatus.WARNING for n in nodes):
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY

        health = ClusterHealth(
            status=overall_status,
            nodes=nodes,
            pods=pods,
            services=services,
            events=events,
        )

        # AI analysis
        if include_ai:
            self.console.print("\n[bold]6. Running AI analysis...[/bold]")
            summary, recommendations = self._analyze_with_ai(health)
            health.summary = summary
            health.recommendations = recommendations

        # Display results
        self._display_health_report(health, verbose)

        return health

    def _display_health_report(self, health: ClusterHealth, verbose: bool = False):
        """Display health report to console.

        Args:
            health: ClusterHealth data
            verbose: Show detailed output
        """
        self.console.print()

        # Status banner
        status_colors = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.WARNING: "yellow",
            HealthStatus.CRITICAL: "red",
            HealthStatus.UNKNOWN: "dim",
        }
        status_icons = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓",
        }

        color = status_colors[health.status]
        icon = status_icons[health.status]

        self.console.print(
            Panel(
                f"[bold {color}]{icon} Cluster Status: {health.status.value.upper()}[/bold {color}]",
                border_style=color,
            )
        )

        # Nodes table
        if health.nodes:
            table = Table(title="Node Status", show_header=True)
            table.add_column("Node", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Message")

            for node in health.nodes:
                status_str = f"[{status_colors[node.status]}]{node.status.value}[/{status_colors[node.status]}]"
                table.add_row(node.name, status_str, node.message[:50])

            self.console.print(table)

        # Unhealthy pods
        if health.pods:
            table = Table(title="Unhealthy Pods", show_header=True)
            table.add_column("Namespace", style="dim")
            table.add_column("Pod", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Issue")

            for pod in health.pods[:15]:  # Limit display
                status_str = f"[{status_colors[pod.status]}]{pod.status.value}[/{status_colors[pod.status]}]"
                table.add_row(pod.namespace, pod.name, status_str, pod.message[:40])

            if len(health.pods) > 15:
                table.add_row("...", f"and {len(health.pods) - 15} more", "", "")

            self.console.print(table)

        # Services without endpoints
        if health.services:
            table = Table(title="Services Without Endpoints", show_header=True)
            table.add_column("Namespace", style="dim")
            table.add_column("Service", style="cyan")
            table.add_column("Issue", style="yellow")

            for svc in health.services[:10]:
                table.add_row(svc.namespace, svc.name, svc.message)

            self.console.print(table)

        # Recent events
        if health.events and verbose:
            table = Table(title="Recent Warning/Error Events", show_header=True)
            table.add_column("Type", style="red")
            table.add_column("Resource", style="cyan")
            table.add_column("Reason", style="yellow")
            table.add_column("Message")

            for event in health.events[:10]:
                resource = f"{event['namespace']}/{event['kind']}/{event['name']}"
                table.add_row(
                    event["type"],
                    resource[:30],
                    event["reason"],
                    event["message"][:40],
                )

            self.console.print(table)

        # AI Summary and Recommendations
        if health.summary:
            self.console.print()
            self.console.print(
                Panel(
                    f"[bold]AI 분석 요약[/bold]\n\n{health.summary}",
                    title="Analysis",
                    border_style="blue",
                )
            )

        if health.recommendations:
            self.console.print()
            rec_text = "\n".join([f"• {rec}" for rec in health.recommendations])
            self.console.print(
                Panel(
                    f"[bold]권장 조치[/bold]\n\n{rec_text}",
                    title="Recommendations",
                    border_style="green",
                )
            )

    def run_node_check(self) -> bool:
        """Quick node status check.

        Returns:
            True if all nodes are healthy
        """
        self.console.print("\n[bold]Node Status:[/bold]")

        success, output = self._run_kubectl("get", "nodes", "-o", "wide")
        if success:
            self.console.print(output)
            return True
        else:
            self.console.print(f"[red]Failed: {output}[/red]")
            return False

    def run_pod_check(self, namespace: str = "") -> bool:
        """Quick pod status check.

        Args:
            namespace: Namespace to check

        Returns:
            True if successful
        """
        ns_args = ["-n", namespace] if namespace else ["-A"]

        self.console.print(f"\n[bold]Pod Status ({namespace or 'all namespaces'}):[/bold]")

        success, output = self._run_kubectl(
            "get", "pods", *ns_args,
            "--field-selector=status.phase!=Running,status.phase!=Succeeded",
        )

        if success:
            if output.strip():
                self.console.print(output)
            else:
                self.console.print("[green]All pods are running or completed[/green]")
            return True
        else:
            self.console.print(f"[red]Failed: {output}[/red]")
            return False

    def run_events(self, namespace: str = "") -> bool:
        """Show recent cluster events.

        Args:
            namespace: Namespace to check

        Returns:
            True if successful
        """
        ns_args = ["-n", namespace] if namespace else ["-A"]

        self.console.print(f"\n[bold]Recent Events ({namespace or 'all namespaces'}):[/bold]")

        success, output = self._run_kubectl(
            "get", "events", *ns_args,
            "--sort-by=.lastTimestamp",
            "--field-selector=type!=Normal",
        )

        if success:
            if output.strip():
                self.console.print(output)
            else:
                self.console.print("[green]No warning/error events[/green]")
            return True
        else:
            self.console.print(f"[red]Failed: {output}[/red]")
            return False

    def run_resources(self, namespace: str = "") -> bool:
        """Show resource usage.

        Args:
            namespace: Namespace to check

        Returns:
            True if successful
        """
        ns_args = ["-n", namespace] if namespace else ["-A"]

        self.console.print("\n[bold]Resource Usage:[/bold]")

        # Node resources
        self.console.print("\n[cyan]Nodes:[/cyan]")
        success, output = self._run_kubectl("top", "nodes")
        if success:
            self.console.print(output)
        else:
            self.console.print("[dim]Node metrics not available (metrics-server required)[/dim]")

        # Pod resources
        self.console.print(f"\n[cyan]Pods ({namespace or 'all namespaces'}):[/cyan]")
        success, output = self._run_kubectl("top", "pods", *ns_args, "--sort-by=memory")
        if success:
            # Show top 20
            lines = output.strip().split("\n")
            self.console.print("\n".join(lines[:21]))
            if len(lines) > 21:
                self.console.print(f"[dim]... and {len(lines) - 21} more pods[/dim]")
        else:
            self.console.print("[dim]Pod metrics not available (metrics-server required)[/dim]")

        return True
