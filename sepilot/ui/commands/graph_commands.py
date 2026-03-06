"""Graph visualization commands for LangGraph workflows.

This module follows the Single Responsibility Principle (SRP) by handling
only graph visualization and structure display.
"""

from typing import Any

from rich.console import Console


def handle_graph_command(
    console: Console,
    agent: Any | None,
    input_text: str,
) -> None:
    """Visualize LangGraph structure as ASCII in console.

    Args:
        console: Rich console for output
        agent: Agent with graph attribute
        input_text: Original command for parsing args
    """
    if not agent:
        console.print("[yellow]⚠️  Agent not available - /graph command disabled[/yellow]")
        return

    try:
        console.print("[bold cyan]📊 LangGraph Workflow Structure[/bold cyan]\n")

        # Get the compiled graph
        graph = agent.graph

        # Parse command arguments
        args = input_text.split()
        xray_mode = '--xray' in args or '-x' in args

        # Extract nodes and edges
        nodes = []
        edges = []

        if hasattr(graph, 'get_graph'):
            try:
                drawable_graph = graph.get_graph(xray=xray_mode) if xray_mode else graph.get_graph()

                if hasattr(drawable_graph, 'nodes'):
                    nodes = list(drawable_graph.nodes.keys())

                if hasattr(drawable_graph, 'edges'):
                    for edge in drawable_graph.edges:
                        if hasattr(edge, 'source') and hasattr(edge, 'target'):
                            edges.append((edge.source, edge.target))
                        elif isinstance(edge, tuple) and len(edge) >= 2:
                            edges.append((edge[0], edge[1]))
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not get detailed graph: {e}[/yellow]")

        # Fallback to basic graph attributes
        if not nodes:
            if hasattr(graph, 'nodes'):
                nodes = list(graph.nodes.keys())
            elif hasattr(graph, '_nodes'):
                nodes = list(graph._nodes.keys())

        if not edges and hasattr(graph, 'edges'):
            raw_edges = graph.edges
            if isinstance(raw_edges, dict):
                for source, targets in raw_edges.items():
                    if isinstance(targets, list):
                        for target in targets:
                            edges.append((source, target))
                    else:
                        edges.append((source, targets))

        # Display ASCII graph
        _print_ascii_graph(console, nodes, edges)

        # Show checkpointer info
        if hasattr(graph, 'checkpointer') and graph.checkpointer:
            checkpointer_type = type(graph.checkpointer).__name__
            console.print(f"\n[cyan]Checkpointer:[/cyan] {checkpointer_type}")

        # Show xray mode hint
        if not xray_mode:
            console.print("\n[dim]💡 Tip: Use '/graph --xray' for detailed internal structure[/dim]")

    except Exception as e:
        console.print(f"[red]Error visualizing graph: {e}[/red]")


def _print_ascii_graph(console: Console, nodes: list, edges: list) -> None:
    """Print graph as ASCII art in console - handles complex graphs with branches and loops.

    Args:
        console: Rich console for output
        nodes: List of node names
        edges: List of (source, target) tuples
    """
    if not nodes:
        console.print("[yellow]No nodes found in graph[/yellow]")
        return

    # Build adjacency lists (forward and backward)
    adj_forward = {node: [] for node in nodes}
    adj_backward = {node: [] for node in nodes}
    for source, target in edges:
        if source in adj_forward:
            adj_forward[source].append(target)
        if target in adj_backward:
            adj_backward[target].append(source)

    # Identify special nodes
    start_node = "__start__" if "__start__" in nodes else None
    end_node = "__end__" if "__end__" in nodes else None

    # Identify branching nodes (multiple outgoing edges)
    branching_nodes = {n for n, t in adj_forward.items() if len(t) > 1}
    # Identify merge nodes (multiple incoming edges)
    merge_nodes = {n for n, t in adj_backward.items() if len(t) > 1}

    # DFS to get node ordering and detect back-edges
    visited_order = []
    visited_set = set()

    def dfs_order(node):
        if node in visited_set or node not in adj_forward:
            return
        visited_set.add(node)
        visited_order.append(node)
        for target in adj_forward.get(node, []):
            dfs_order(target)

    start = start_node if start_node else (nodes[0] if nodes else None)
    if start:
        dfs_order(start)

    # Add any unvisited nodes
    for n in nodes:
        if n not in visited_set:
            visited_order.append(n)

    node_index = {n: i for i, n in enumerate(visited_order)}
    back_edges = []
    for source, target in edges:
        if source in node_index and target in node_index:
            if node_index[source] >= node_index[target] and target != end_node:
                back_edges.append((source, target))

    loop_targets = {target for source, target in back_edges}

    # Filter to real nodes only (exclude __start__, __end__)
    real_nodes = [n for n in nodes if not n.startswith("__")]

    # Print summary
    console.print("[bold green]Graph Summary:[/bold green]")
    console.print(f"  Nodes: {len(real_nodes)} | Edges: {len(edges)} | Branches: {len(branching_nodes)} | Loops: {len(back_edges)}")

    # Print all edges in a clear format
    console.print("\n[bold cyan]All Edges:[/bold cyan]")
    for source, target in edges:
        src_display = "START" if source == "__start__" else source
        tgt_display = "END" if target == "__end__" else target

        # Determine edge type
        if (source, target) in back_edges:
            console.print(f"  [red]↺[/red] {src_display} [red]──loop──→[/red] {tgt_display}")
        elif source in branching_nodes:
            console.print(f"  [yellow]◇[/yellow] {src_display} [yellow]──branch──→[/yellow] {tgt_display}")
        else:
            console.print(f"  [cyan]●[/cyan] {src_display} [cyan]──→[/cyan] {tgt_display}")

    # Print detailed node connections
    console.print("\n[bold green]Node Details:[/bold green]")
    for node in visited_order:
        if node.startswith("__"):
            continue

        # Determine node characteristics
        indicators = []
        if node in branching_nodes:
            indicators.append("[yellow]◇ branch[/yellow]")
        if node in merge_nodes:
            indicators.append("[magenta]⊕ merge[/magenta]")
        if node in loop_targets:
            indicators.append("[red]↺ loop-target[/red]")

        indicator_str = " ".join(indicators) if indicators else ""

        targets = adj_forward.get(node, [])
        sources = adj_backward.get(node, [])

        # Format incoming
        src_display = [("START" if s == "__start__" else s) for s in sources]
        # Format outgoing
        tgt_display = [("END" if t == "__end__" else t) for t in targets]

        console.print(f"  [bold white]{node}[/bold white] {indicator_str}")
        if src_display:
            console.print(f"    [dim]← from:[/dim] {', '.join(src_display)}")
        if tgt_display:
            console.print(f"    [dim]→ to:[/dim]   {', '.join(tgt_display)}")

    # Print visual flow - text-based representation
    console.print("\n[bold green]Workflow Flow (text representation):[/bold green]")

    # Show flow as a tree-like structure
    def print_flow_from(node, prefix="", visited=None, depth=0):
        if visited is None:
            visited = set()
        if depth > 15:  # Prevent infinite recursion
            return
        if node in visited:
            console.print(f"{prefix}[red]↺ (back to {node})[/red]")
            return

        visited.add(node)
        targets = adj_forward.get(node, [])

        # Display current node
        if node == "__start__":
            console.print(f"{prefix}[green]▶ START[/green]")
        elif node == "__end__":
            console.print(f"{prefix}[red]■ END[/red]")
        elif node in branching_nodes:
            console.print(f"{prefix}[yellow]◇ {node}[/yellow]")
        elif node in loop_targets:
            console.print(f"{prefix}[magenta]↺ {node}[/magenta]")
        else:
            console.print(f"{prefix}[cyan]● {node}[/cyan]")

        # Process children
        for i, target in enumerate(targets):
            is_last = (i == len(targets) - 1)
            connector = "└─→ " if is_last else "├─→ "
            prefix + ("    " if is_last else "│   ")

            if target in visited and target != "__end__":
                # Back edge
                console.print(f"{prefix}{connector}[red]↺ (loop back to {target})[/red]")
            else:
                print_flow_from(target, prefix + connector, visited.copy(), depth + 1)

    if start_node:
        print_flow_from(start_node)

    # Show back-edges summary
    if back_edges:
        console.print("\n[bold red]Loop Edges (back-edges):[/bold red]")
        for source, target in back_edges:
            console.print(f"  [red]↺[/red] {source} [red]──loops back to──→[/red] {target}")

    # Legend
    console.print("\n[dim]Legend: ◇ conditional branch | ↺ loop target | ⊕ merge point[/dim]")


def show_basic_graph_info(console: Console, graph: Any) -> None:
    """Fallback method to show basic graph information.

    Args:
        console: Rich console for output
        graph: LangGraph compiled graph object
    """
    console.print("[bold yellow]Basic Graph Information:[/bold yellow]")

    # Try to extract nodes
    nodes = []
    if hasattr(graph, 'nodes'):
        nodes = list(graph.nodes.keys())
    elif hasattr(graph, '_nodes'):
        nodes = list(graph._nodes.keys())

    if nodes:
        console.print(f"[cyan]Nodes ({len(nodes)}):[/cyan]")
        for node in nodes:
            console.print(f"  • {node}")

    # Try to extract edges
    edges = []
    if hasattr(graph, 'edges'):
        edges = graph.edges
    elif hasattr(graph, '_edges'):
        edges = graph._edges

    if edges:
        console.print("\n[cyan]Edges:[/cyan]")
        if isinstance(edges, dict):
            for source, targets in edges.items():
                if isinstance(targets, list):
                    for target in targets:
                        console.print(f"  {source} → {target}")
                else:
                    console.print(f"  {source} → {targets}")

    console.print("\n[dim]💡 This graph doesn't support advanced visualization[/dim]")
