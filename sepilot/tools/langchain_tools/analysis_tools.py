"""Code analysis tools for LangChain agent."""


from langchain_core.tools import tool

# Singleton analyzer instance
_code_analyzer = None


def _get_code_analyzer():
    """Get or create CodeAnalyzer singleton."""
    global _code_analyzer
    if _code_analyzer is None:
        from sepilot.tools.code_analysis.analyzer import CodeAnalyzer
        _code_analyzer = CodeAnalyzer()
    return _code_analyzer


@tool
def code_analyze(
    action: str,
    file_path: str | None = None,
    function_name: str | None = None,
    class_name: str | None = None,
    symbol_name: str | None = None,
    directory: str | None = None,
    threshold: int | None = 10,
    format: str = "text"
) -> str:
    """Analyze Python code structure using AST (Abstract Syntax Tree).

    This tool can extract functions, classes, imports, calculate complexity,
    and find symbol references in Python code.

    Args:
        action: Analysis action to perform:
            - "analyze_file": Get complete file analysis (requires file_path)
            - "list_functions": List all functions in a file (requires file_path)
            - "list_classes": List all classes in a file (requires file_path)
            - "find_function": Analyze specific function (requires file_path, function_name)
            - "find_class": Analyze specific class (requires file_path, class_name)
            - "analyze_imports": Analyze import statements (requires file_path)
            - "check_complexity": Check functions exceeding complexity threshold (requires file_path, threshold)
            - "find_references": Find all references to a symbol (requires symbol_name, directory)
        file_path: Path to the Python file to analyze
        function_name: Name of the function to find
        class_name: Name of the class to find
        symbol_name: Name of the symbol to find references for
        directory: Directory to search for references (default: ".")
        threshold: Complexity threshold for check_complexity (default: 10)
        format: Output format - "text" or "json" (default: "text")

    Returns:
        Analysis result as formatted text or JSON

    Examples:
        code_analyze(action="analyze_file", file_path="main.py")
        code_analyze(action="list_functions", file_path="utils.py")
        code_analyze(action="find_function", file_path="calculator.py", function_name="add")
        code_analyze(action="check_complexity", file_path="main.py", threshold=10)
        code_analyze(action="find_references", symbol_name="UserManager", directory=".")
    """
    try:
        analyzer = _get_code_analyzer()

        if action == "analyze_file":
            if not file_path:
                return "Error: file_path is required for analyze_file action"
            analyzer.analyze_file(file_path)
            return analyzer.format_file_analysis(file_path, format=format)

        elif action == "list_functions":
            if not file_path:
                return "Error: file_path is required for list_functions action"
            functions = analyzer.list_functions(file_path)
            if format == "json":
                import json
                return json.dumps({"functions": functions}, indent=2)
            result = f"📋 Functions in {file_path}:\n"
            for func in functions:
                result += f"  • {func}\n"
            return result

        elif action == "list_classes":
            if not file_path:
                return "Error: file_path is required for list_classes action"
            classes = analyzer.list_classes(file_path)
            if format == "json":
                import json
                return json.dumps({"classes": classes}, indent=2)
            result = f"📦 Classes in {file_path}:\n"
            for cls in classes:
                result += f"  • {cls}\n"
            return result

        elif action == "find_function":
            if not file_path or not function_name:
                return "Error: file_path and function_name are required for find_function action"
            func_info = analyzer.find_function(file_path, function_name)
            if not func_info:
                return f"Error: Function '{function_name}' not found in {file_path}"
            return analyzer.format_function_info(func_info, format=format)

        elif action == "find_class":
            if not file_path or not class_name:
                return "Error: file_path and class_name are required for find_class action"
            class_info = analyzer.find_class(file_path, class_name)
            if not class_info:
                return f"Error: Class '{class_name}' not found in {file_path}"
            return analyzer.format_class_info(class_info, format=format)

        elif action == "analyze_imports":
            if not file_path:
                return "Error: file_path is required for analyze_imports action"
            import_info = analyzer.analyze_imports(file_path)
            if format == "json":
                import json
                return json.dumps(import_info.to_dict(), indent=2)

            result = f"📚 Imports in {file_path}:\n\n"
            if import_info.imports:
                result += "Standard imports:\n"
                for imp in import_info.imports:
                    alias_str = f" as {imp.alias}" if imp.alias else ""
                    result += f"  • import {imp.module}{alias_str}\n"
            if import_info.from_imports:
                result += "\nFrom imports:\n"
                for from_imp in import_info.from_imports:
                    names = ", ".join([f"{n['name']}" + (f" as {n['alias']}" if n['alias'] else "")
                                      for n in from_imp.names])
                    result += f"  • from {from_imp.module} import {names}\n"
            external_deps = import_info.get_external_dependencies()
            if external_deps:
                result += "\nExternal dependencies:\n"
                for dep in external_deps:
                    result += f"  • {dep}\n"
            return result

        elif action == "check_complexity":
            if not file_path:
                return "Error: file_path is required for check_complexity action"
            complex_functions = analyzer.check_complexity(file_path, threshold=threshold)
            if format == "json":
                import json
                return json.dumps(complex_functions, indent=2)
            if not complex_functions:
                return f"✅ No functions exceed complexity threshold of {threshold} in {file_path}"
            result = f"⚠️  Functions exceeding complexity threshold ({threshold}) in {file_path}:\n\n"
            for func in complex_functions:
                result += f"• {func['name']} ({func['type']}) [line {func['line']}]\n"
                metrics = func['metrics']
                result += f"  Complexity: {metrics['cyclomatic_complexity']}\n"
                result += f"  Lines: {metrics['lines_of_code']}\n"
                result += f"  Nesting depth: {metrics['max_nesting_depth']}\n"
                result += f"  Parameters: {metrics['parameter_count']}\n"
                if metrics.get('suggestions'):
                    result += "  Suggestions:\n"
                    for suggestion in metrics['suggestions']:
                        result += f"    - {suggestion}\n"
                result += "\n"
            return result

        elif action == "find_references":
            if not symbol_name:
                return "Error: symbol_name is required for find_references action"
            search_dir = directory or "."
            references = analyzer.find_references(symbol_name, search_dir)
            if format == "json":
                import json
                return json.dumps([
                    {
                        "file_path": ref.file_path,
                        "line_number": ref.line_number,
                        "context": ref.context,
                        "type": ref.reference_type
                    }
                    for ref in references
                ], indent=2)
            if not references:
                return f"No references found for '{symbol_name}' in {search_dir}"
            result = f"🔍 References to '{symbol_name}' in {search_dir}:\n\n"
            by_file = {}
            for ref in references:
                if ref.file_path not in by_file:
                    by_file[ref.file_path] = []
                by_file[ref.file_path].append(ref)
            for fp, refs in sorted(by_file.items()):
                result += f"📄 {fp}:\n"
                for ref in refs:
                    result += f"  Line {ref.line_number} [{ref.reference_type}]: {ref.context}\n"
                result += "\n"
            result += f"Total: {len(references)} reference(s) found"
            return result

        else:
            return f"Error: Unknown action '{action}'. Valid actions: analyze_file, list_functions, list_classes, find_function, find_class, analyze_imports, check_complexity, find_references"

    except FileNotFoundError as e:
        return f"Error: {str(e)}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error analyzing code: {str(e)}"


__all__ = ['code_analyze', '_get_code_analyzer']
