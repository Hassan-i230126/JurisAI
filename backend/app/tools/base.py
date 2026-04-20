"""
Juris AI — Tool Base Classes
Abstract base class for all tools and the ToolResult dataclass.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from app.models.schemas import ToolResult


class ToolBase(ABC):
    """
    Abstract base class for all Juris AI tools.
    
    Every tool must define a name, description, input schema,
    and implement the async run() method.
    """

    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Tool-specific arguments matching the input_schema.
            
        Returns:
            ToolResult with success status, data, and formatted text.
        """
        pass

    def get_tool_definition(self) -> dict:
        """
        Return a JSON-serializable definition of this tool.
        
        Used for documentation and LLM tool-calling schema injection.
        """
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.input_schema,
        }
