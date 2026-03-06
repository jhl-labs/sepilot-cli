"""Adapter to convert our BaseTool to LangChain StructuredTool"""


from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from sepilot.tools.base_tool import BaseTool


def create_pydantic_model(tool: BaseTool) -> type[BaseModel]:
    """Create a Pydantic model from tool parameters"""
    fields = {}

    for param_name, param_desc in tool.parameters.items():
        # Parse description to determine if required
        is_required = "required" in param_desc.lower()

        # Create field with description
        if is_required:
            fields[param_name] = (str, Field(..., description=param_desc))
        else:
            fields[param_name] = (str, Field(None, description=param_desc))

    # Create dynamic Pydantic model
    model_name = f"{tool.name.title()}Input"
    return create_model(model_name, **fields)


def convert_to_langchain_tool(tool: BaseTool) -> StructuredTool:
    """Convert our BaseTool to LangChain StructuredTool"""

    # Create input schema
    input_schema = create_pydantic_model(tool)

    # Create the tool
    return StructuredTool.from_function(
        func=tool.execute,
        name=tool.name,
        description=tool.description,
        args_schema=input_schema,
        return_direct=False
    )


def convert_all_tools(tools: list[BaseTool]) -> list[StructuredTool]:
    """Convert all tools to LangChain format"""
    return [convert_to_langchain_tool(tool) for tool in tools]
