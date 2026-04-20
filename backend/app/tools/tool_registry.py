"""
Juris AI — Tool Registry
Registry pattern for discovering and managing all available tools.
"""

from typing import Dict, List, Optional

from loguru import logger

from app.tools.base import ToolBase


class ToolRegistry:
    """
    Central registry for all Juris AI tools.
    
    Implements the registry pattern: tools register themselves,
    and the orchestrator discovers them by name at runtime.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, ToolBase] = {}

    def register(self, tool: ToolBase) -> None:
        """
        Register a tool instance.
        
        Args:
            tool: An instance of a ToolBase subclass.
            
        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info("Registered tool: {} — {}", tool.name, tool.description)

    def get(self, name: str) -> Optional[ToolBase]:
        """
        Retrieve a registered tool by name.
        
        Args:
            name: The tool name to look up.
            
        Returns:
            The tool instance, or None if not found.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[dict]:
        """
        Return definitions of all registered tools.
        
        Returns:
            List of tool definition dictionaries.
        """
        return [tool.get_tool_definition() for tool in self._tools.values()]

    def get_all_names(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    @property
    def count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)


# Global registry instance — populated during app startup
registry = ToolRegistry()


def register_all_tools(db_path: str, chroma_collection=None) -> ToolRegistry:
    """
    Instantiate and register all four Juris AI tools.
    
    Called during FastAPI lifespan startup to make tools available
    to the conversation manager and orchestrator.
    
    Args:
        db_path: Path to the SQLite database.
        chroma_collection: ChromaDB collection for case search.
        
    Returns:
        The populated ToolRegistry.
    """
    from app.tools.crm_tool import CRMTool
    from app.tools.statute_lookup_tool import StatuteLookupTool
    from app.tools.case_search_tool import CaseSearchTool
    from app.tools.deadline_calc_tool import DeadlineCalcTool

    global registry
    registry = ToolRegistry()

    # Tool 1: CRM — client profile management
    crm = CRMTool(db_path=db_path)
    registry.register(crm)

    # Tool 2: Statute Lookup — PPC/CrPC section lookup
    statute = StatuteLookupTool(db_path=db_path)
    registry.register(statute)

    # Tool 3: Case Search — Supreme Court judgment semantic search
    case_search = CaseSearchTool(chroma_collection=chroma_collection)
    registry.register(case_search)

    # Tool 4: Deadline Calculator — CrPC-based legal deadline computation
    deadline = DeadlineCalcTool()
    registry.register(deadline)

    logger.info("All {} tools registered successfully", registry.count)
    return registry
