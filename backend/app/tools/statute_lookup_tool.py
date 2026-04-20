"""
Juris AI — Statute Lookup Tool
PPC/CrPC/QSO/ATA section lookup by number or keyword.
Uses SQLite with FTS5 full-text search for fast keyword queries.
"""

from typing import Optional, List

import aiosqlite
from loguru import logger

from app.tools.base import ToolBase
from app.models.schemas import ToolResult


class StatuteLookupTool(ToolBase):
    """
    Looks up Pakistani criminal law statute sections by number or keyword.
    
    Backed by a SQLite database with a statutes table containing
    section text, punishments, classification (cognizable/bailable),
    and the court with jurisdiction.
    """

    name = "statute_lookup"
    description = "Look up a specific section of PPC, CrPC, QSO, or ATA by section number or search by keyword."
    input_schema = {
        "type": "object",
        "properties": {
            "act": {
                "type": "string",
                "enum": ["PPC", "CrPC", "QSO", "ATA"],
                "description": "The act to look up (PPC, CrPC, QSO, or ATA)"
            },
            "section_number": {
                "type": "string",
                "description": "The section number to look up (e.g. '302', '496-B')"
            },
            "keyword": {
                "type": "string",
                "description": "Keyword to search across section titles and text"
            }
        }
    }

    def __init__(self, db_path: str):
        """
        Initialize the statute lookup tool.
        
        Args:
            db_path: Path to the SQLite database containing the statutes table.
        """
        self.db_path = db_path

    async def initialize_schema(self) -> None:
        """Create the statutes table and FTS5 virtual table if they don't exist."""
        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS statutes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    act TEXT NOT NULL,
                    section_number TEXT,
                    title TEXT,
                    text TEXT,
                    punishment TEXT,
                    cognizability TEXT,
                    bailable TEXT,
                    triable_by TEXT
                )
            """)
            # Create FTS5 virtual table for full-text search
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS statutes_fts USING fts5(
                    title, text, content='statutes', content_rowid='id'
                )
            """)
            await db.commit()
            logger.info("Statutes database schema initialized")

    async def run(self, **kwargs) -> ToolResult:
        """
        Execute a statute lookup.
        
        Args:
            act: The act code (PPC, CrPC, QSO, ATA).
            section_number: Specific section number to look up.
            keyword: Keyword for full-text search.
            
        Returns:
            ToolResult with the section details or search results.
        """
        act = kwargs.get("act", "")
        section_number = kwargs.get("section_number", "")
        keyword = kwargs.get("keyword", "")

        try:
            if section_number:
                return await self._lookup_by_section(act, section_number)
            elif keyword:
                return await self._search_by_keyword(keyword)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    formatted_text="Please provide either a section number or a keyword to search.",
                    error_message="Missing section_number or keyword"
                )
        except Exception as e:
            logger.error("Statute lookup error: {}", str(e))
            return ToolResult(
                success=False,
                data=None,
                formatted_text=f"Statute lookup failed: {str(e)}",
                error_message=str(e)
            )

    async def _lookup_by_section(self, act: str, section_number: str) -> ToolResult:
        """Look up a specific statute section by act and section number."""
        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            db.row_factory = aiosqlite.Row

            if act:
                cursor = await db.execute(
                    "SELECT * FROM statutes WHERE act = ? AND section_number = ?",
                    (act.upper(), section_number)
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM statutes WHERE section_number = ?",
                    (section_number,)
                )

            row = await cursor.fetchone()

        if not row:
            not_found_msg = (
                f"Section {section_number}"
                + (f" of {act}" if act else "")
                + " not found in database. Please refer to the official Pakistan Code at pakistancode.gov.pk"
            )
            return ToolResult(
                success=False,
                data=None,
                formatted_text=not_found_msg,
                error_message="Section not found"
            )

        section = dict(row)
        formatted = self._format_section(section)
        return ToolResult(success=True, data=section, formatted_text=formatted)

    async def _search_by_keyword(self, keyword: str) -> ToolResult:
        """Search statutes by keyword using FTS5 full-text search."""
        async with aiosqlite.connect(self.db_path, timeout=10) as db:
            db.row_factory = aiosqlite.Row

            # Try FTS5 search first
            try:
                cursor = await db.execute(
                    """SELECT s.* FROM statutes s 
                       JOIN statutes_fts fts ON s.id = fts.rowid 
                       WHERE statutes_fts MATCH ? 
                       LIMIT 5""",
                    (keyword,)
                )
                rows = await cursor.fetchall()
            except Exception:
                # Fallback to LIKE search if FTS5 fails
                cursor = await db.execute(
                    """SELECT * FROM statutes 
                       WHERE title LIKE ? OR text LIKE ? 
                       LIMIT 5""",
                    (f"%{keyword}%", f"%{keyword}%")
                )
                rows = await cursor.fetchall()

        if not rows:
            return ToolResult(
                success=True,
                data=[],
                formatted_text=f"No statute sections found matching '{keyword}'. Try a different search term or refer to pakistancode.gov.pk"
            )

        sections = [dict(row) for row in rows]
        lines = [f"**Statute search results for '{keyword}':**\n"]
        for section in sections:
            lines.append(self._format_section(section))
            lines.append("---")

        return ToolResult(
            success=True,
            data=sections,
            formatted_text="\n".join(lines)
        )

    @staticmethod
    def _format_section(section: dict) -> str:
        """Format a statute section dictionary as a readable string."""
        act = section.get("act", "")
        sec_num = section.get("section_number", "")
        title = section.get("title", "")
        text = section.get("text", "")
        punishment = section.get("punishment", "")
        cognizability = section.get("cognizability", "")
        bailable = section.get("bailable", "")
        triable_by = section.get("triable_by", "")

        header = f"**{act} Section {sec_num}**"
        if title:
            header += f" — {title}"

        lines = [header]
        if text:
            # Truncate very long text for display
            display_text = text[:800] + "..." if len(text) > 800 else text
            lines.append(f"\n{display_text}")
        if punishment:
            lines.append(f"\n**Punishment:** {punishment}")
        if cognizability:
            lines.append(f"**Cognizability:** {cognizability}")
        if bailable:
            lines.append(f"**Bailable:** {bailable}")
        if triable_by:
            lines.append(f"**Triable by:** {triable_by}")

        return "\n".join(lines)
