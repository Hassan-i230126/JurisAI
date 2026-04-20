"""
Juris AI — CRM Tool
Client Relationship Management for criminal defense cases.
Stores client profiles and interaction logs in SQLite via aiosqlite.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

import aiosqlite
from loguru import logger

from app.tools.base import ToolBase
from app.models.schemas import ToolResult


class CRMTool(ToolBase):
    """
    CRM tool for managing client profiles and interaction history.
    
    Supports creating, reading, updating, listing, and searching
    client records. All data is stored in SQLite.
    """

    name = "crm_tool"
    description = "Manage client profiles: create, view, update, list, and search criminal defense clients."
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "get", "update", "list", "search", "log_interaction"],
                "description": "The CRM operation to perform"
            },
            "client_id": {"type": "string", "description": "Client ID for get/update operations"},
            "name": {"type": "string", "description": "Client name for create/search"},
            "cnic": {"type": "string", "description": "Client CNIC number"},
            "contact": {"type": "string", "description": "Client contact number"},
            "case_type": {"type": "string", "description": "Type of criminal case"},
            "charges": {"type": "string", "description": "Comma-separated PPC sections"},
            "bail_status": {"type": "string", "description": "on bail | in custody | acquitted | convicted | unknown"},
            "court_name": {"type": "string", "description": "Name of the court"},
            "next_hearing_date": {"type": "string", "description": "Next hearing date (YYYY-MM-DD)"},
            "notes": {"type": "string", "description": "Additional notes"},
            "field": {"type": "string", "description": "Field name to update (name, cnic, contact, case_type, charges, bail_status, court_name, next_hearing_date, notes)"},
            "value": {"type": "string", "description": "New value for the field"},
            "query": {"type": "string", "description": "Search query (name or CNIC)"},
            "session_id": {"type": "string", "description": "Session ID for interaction logging"},
            "summary": {"type": "string", "description": "Interaction summary"},
        },
        "required": ["action"]
    }

    def __init__(self, db_path: str):
        """
        Initialize the CRM tool with the SQLite database path.
        
        Args:
            db_path: Absolute path to the SQLite database file.
        """
        self.db_path = db_path

    async def initialize_schema(self) -> None:
        """Create the clients and interactions tables if they don't exist."""
        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS clients (
                    client_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    cnic TEXT,
                    contact TEXT,
                    case_type TEXT,
                    charges TEXT,
                    bail_status TEXT,
                    court_name TEXT,
                    next_hearing_date TEXT,
                    notes TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT,
                    session_id TEXT,
                    summary TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (client_id) REFERENCES clients(client_id)
                )
            """)
            await db.commit()
            logger.info("CRM database schema initialized")

    async def run(self, **kwargs) -> ToolResult:
        """
        Execute a CRM operation.
        
        Args:
            **kwargs: Operation-specific arguments (action is required).
            
        Returns:
            ToolResult with the operation outcome.
        """
        action = kwargs.get("action", "")

        try:
            if action == "create":
                return await self._create_client(**kwargs)
            elif action == "get":
                return await self._get_client(kwargs.get("client_id", ""))
            elif action == "update":
                return await self._update_client(
                    kwargs.get("client_id", ""),
                    kwargs.get("field", ""),
                    kwargs.get("value", "")
                )
            elif action == "list":
                return await self._list_clients()
            elif action == "delete":
                return await self._delete_client(kwargs.get("client_id", ""))
            elif action == "search":
                return await self._search_clients(kwargs.get("query", kwargs.get("name", "")))
            elif action == "log_interaction":
                return await self._log_interaction(
                    kwargs.get("client_id", ""),
                    kwargs.get("session_id", ""),
                    kwargs.get("summary", "")
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    formatted_text=f"Unknown CRM action: '{action}'. Available: create, get, update, list, search, log_interaction.",
                    error_message=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error("CRM tool error | action={} | error={}", action, str(e))
            return ToolResult(
                success=False,
                data=None,
                formatted_text=f"CRM operation failed: {str(e)}",
                error_message=str(e)
            )

    async def _create_client(self, **kwargs) -> ToolResult:
        """Create a new client profile."""
        client_id = kwargs.get("client_id")
        if not client_id:
            client_id = str(uuid.uuid4())[:8]

        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            # Check if client_id already exists
            cursor = await db.execute("SELECT 1 FROM clients WHERE client_id = ?", (client_id,))
            if await cursor.fetchone():
                return ToolResult(
                    success=False,
                    data=None,
                    formatted_text=f"Client ID '{client_id}' is already taken.",
                    error_message=f"Client ID '{client_id}' is already taken."
                )

            await db.execute(
                """INSERT INTO clients 
                   (client_id, name, cnic, contact, case_type, charges, 
                    bail_status, court_name, next_hearing_date, notes, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    client_id,
                    kwargs.get("name", "Unknown"),
                    kwargs.get("cnic"),
                    kwargs.get("contact"),
                    kwargs.get("case_type"),
                    kwargs.get("charges"),
                    kwargs.get("bail_status", "unknown"),
                    kwargs.get("court_name"),
                    kwargs.get("next_hearing_date"),
                    kwargs.get("notes"),
                    now,
                    now,
                )
            )
            await db.commit()

        logger.info("Client created | id={} | name={}", client_id, kwargs.get("name"))
        return ToolResult(
            success=True,
            data={"client_id": client_id},
            formatted_text=f"Client profile created successfully.\nClient ID: {client_id}\nName: {kwargs.get('name', 'Unknown')}"
        )

    async def _get_client(self, client_id: str) -> ToolResult:
        """Retrieve a client profile by ID."""
        if not client_id:
            return ToolResult(
                success=False, data=None,
                formatted_text="Please provide a client ID.",
                error_message="Missing client_id"
            )

        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM clients WHERE client_id = ?", (client_id,)
            )
            row = await cursor.fetchone()

        if not row:
            return ToolResult(
                success=False, data=None,
                formatted_text=f"No client found with ID: {client_id}",
                error_message="Client not found"
            )

        client = dict(row)
        formatted = self._format_client_profile(client)
        return ToolResult(success=True, data=client, formatted_text=formatted)

    async def _update_client(self, client_id: str, field: str, value: str) -> ToolResult:
        """Update a specific field of a client profile."""
        allowed_fields = [
            "name", "cnic", "contact", "case_type", "charges",
            "bail_status", "court_name", "next_hearing_date", "notes"
        ]

        if not client_id:
            return ToolResult(
                success=False, data=None,
                formatted_text="Please provide a client ID.",
                error_message="Missing client_id"
            )

        if field not in allowed_fields:
            return ToolResult(
                success=False, data=None,
                formatted_text=f"Cannot update field '{field}'. Allowed fields: {', '.join(allowed_fields)}",
                error_message=f"Invalid field: {field}"
            )

        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            await db.execute(
                f"UPDATE clients SET {field} = ?, updated_at = ? WHERE client_id = ?",
                (value, now, client_id)
            )
            await db.commit()

        logger.info("Client updated | id={} | field={}", client_id, field)
        return ToolResult(
            success=True,
            data={"client_id": client_id, "field": field, "value": value},
            formatted_text=f"Client {client_id} updated: {field} = {value}"
        )

    async def _list_clients(self) -> ToolResult:
        """List all client profiles."""
        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT client_id, name, charges, bail_status FROM clients ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()

        if not rows:
            return ToolResult(
                success=True, data=[],
                formatted_text="No clients found in the system."
            )

        clients = [dict(row) for row in rows]
        lines = ["**Clients:**"]
        for c in clients:
            status = c.get("bail_status", "unknown")
            charges = c.get("charges", "N/A")
            lines.append(f"• {c['name']} (ID: {c['client_id']}) — Charges: {charges} | Status: {status}")

        return ToolResult(
            success=True,
            data=clients,
            formatted_text="\n".join(lines)
        )

    async def _delete_client(self, client_id: str) -> ToolResult:
        """Delete a client profile."""
        if not client_id:
            return ToolResult(
                success=False, data=None,
                formatted_text="Please provide a client ID to delete.",
                error_message="Missing client_id"
            )

        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            await db.execute("DELETE FROM interactions WHERE client_id = ?", (client_id,))
            cursor = await db.execute("DELETE FROM clients WHERE client_id = ?", (client_id,))
            await db.commit()
            if cursor.rowcount == 0:
                return ToolResult(
                    success=False, data=None,
                    formatted_text=f"No client found with ID '{client_id}'.",
                    error_message="Client not found"
                )

        import logging
        logging.getLogger(__name__).info(f"Client deleted | id={client_id}")
        return ToolResult(
            success=True,
            data={"client_id": client_id},
            formatted_text=f"Client '{client_id}' successfully deleted."
        )

    async def _search_clients(self, query: str) -> ToolResult:
        """Search clients by name or CNIC."""
        if not query:
            return await self._list_clients()

        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM clients WHERE name LIKE ? OR cnic LIKE ?",
                (f"%{query}%", f"%{query}%")
            )
            rows = await cursor.fetchall()

        if not rows:
            return ToolResult(
                success=True, data=[],
                formatted_text=f"No clients found matching '{query}'."
            )

        clients = [dict(row) for row in rows]
        lines = [f"**Search results for '{query}':**"]
        for c in clients:
            lines.append(self._format_client_profile(c))

        return ToolResult(
            success=True,
            data=clients,
            formatted_text="\n\n".join(lines)
        )

    async def _log_interaction(self, client_id: str, session_id: str, summary: str) -> ToolResult:
        """Log an interaction with a client."""
        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            await db.execute(
                "INSERT INTO interactions (client_id, session_id, summary, timestamp) VALUES (?, ?, ?, ?)",
                (client_id, session_id, summary, now)
            )
            await db.commit()

        return ToolResult(
            success=True,
            data={"client_id": client_id, "session_id": session_id},
            formatted_text=f"Interaction logged for client {client_id}."
        )

    async def get_client_context(self, client_id: str) -> Optional[str]:
        """
        Get a formatted client profile string for injection into the system prompt.
        
        Args:
            client_id: The client ID to look up.
            
        Returns:
            A formatted context string, or None if client not found.
        """
        result = await self._get_client(client_id)
        if result.success and result.data:
            c = result.data
            return (
                f"[CLIENT PROFILE] Client ID: {c.get('client_id', 'N/A')} | "
                f"Name: {c.get('name', 'N/A')} | "
                f"Contact: {c.get('contact', 'N/A')} | "
                f"Case Type: {c.get('case_type', 'N/A')} | "
                f"Charges: {c.get('charges', 'N/A')} | "
                f"Bail Status: {c.get('bail_status', 'N/A')} | "
                f"Court: {c.get('court_name', 'N/A')} | "
                f"Next Hearing: {c.get('next_hearing_date', 'N/A')} | "
                f"Notes: {c.get('notes', 'N/A')}"
            )
        return None

    @staticmethod
    def _format_client_profile(client: dict) -> str:
        """Format a client dictionary as a readable string."""
        return (
            f"**{client.get('name', 'Unknown')}** (ID: {client.get('client_id', 'N/A')})\n"
            f"  CNIC: {client.get('cnic', 'N/A')}\n"
            f"  Contact: {client.get('contact', 'N/A')}\n"
            f"  Case Type: {client.get('case_type', 'N/A')}\n"
            f"  Charges: {client.get('charges', 'N/A')}\n"
            f"  Bail Status: {client.get('bail_status', 'N/A')}\n"
            f"  Court: {client.get('court_name', 'N/A')}\n"
            f"  Next Hearing: {client.get('next_hearing_date', 'N/A')}\n"
            f"  Notes: {client.get('notes', 'N/A')}"
        )
