"""
Juris AI — Legal Deadline Calculator Tool
Pure Python rules engine for CrPC-based legal deadline computation.
No network calls — entirely local computation.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger

from app.tools.base import ToolBase
from app.models.schemas import ToolResult, DeadlineResult


# ─── CrPC Deadline Rules ─────────────────────────────────────────────────────
DEADLINE_RULES = {
    "bail_application_sessions": {
        "description": "Bail application to Sessions Court after arrest",
        "trigger": "arrest",
        "days": 15,
        "authority": "CrPC Section 497",
        "note": "Must file within 15 days if Magistrate refused bail"
    },
    "bail_application_high_court": {
        "description": "Bail application to High Court",
        "trigger": "sessions_court_bail_refusal",
        "days": 30,
        "authority": "CrPC Section 498",
        "note": ""
    },
    "appeal_acquittal_state": {
        "description": "State appeal against acquittal",
        "trigger": "acquittal_order",
        "days": 30,
        "authority": "CrPC Section 417",
        "note": ""
    },
    "appeal_conviction": {
        "description": "Convicted accused appeal against conviction",
        "trigger": "conviction_order",
        "days": 30,
        "authority": "CrPC Section 410",
        "note": ""
    },
    "revision_petition": {
        "description": "Revision petition to High Court",
        "trigger": "order_date",
        "days": 90,
        "authority": "CrPC Section 435",
        "note": ""
    },
    "challan_submission": {
        "description": "Police must submit challan/investigation report",
        "trigger": "arrest",
        "days": 14,
        "authority": "CrPC Section 173",
        "note": "14 days from arrest for non-terrorism cases"
    },
    "bail_extension_remand": {
        "description": "Maximum remand period without charge",
        "trigger": "arrest",
        "days": 15,
        "authority": "CrPC Section 167",
        "note": "15 days maximum — may be extended by court"
    },
}


class DeadlineCalcTool(ToolBase):
    """
    Calculates legal deadlines based on CrPC procedural timelines.
    
    Takes a trigger event and a date, then computes all applicable
    deadlines from the DEADLINE_RULES dictionary.
    """

    name = "deadline_calculator"
    description = "Calculate legal deadlines and time limits based on CrPC procedural rules given a trigger event and date."
    input_schema = {
        "type": "object",
        "properties": {
            "trigger_event": {
                "type": "string",
                "description": "The triggering event: arrest, sessions_court_bail_refusal, acquittal_order, conviction_order, order_date",
                "enum": ["arrest", "sessions_court_bail_refusal", "acquittal_order", "conviction_order", "order_date"]
            },
            "event_date": {
                "type": "string",
                "description": "The date of the trigger event (YYYY-MM-DD or DD/MM/YYYY)"
            }
        },
        "required": ["trigger_event"]
    }

    async def run(self, **kwargs) -> ToolResult:
        """
        Calculate deadlines for the given trigger event.
        
        Args:
            trigger_event: The type of event that triggers deadlines.
            event_date: The date of the event (optional — if missing, uses today).
            
        Returns:
            ToolResult with computed deadlines.
        """
        trigger_event = kwargs.get("trigger_event", "")
        event_date_str = kwargs.get("event_date", "")

        if not trigger_event:
            # If no trigger specified, show all possible triggers
            triggers = sorted(set(rule["trigger"] for rule in DEADLINE_RULES.values()))
            return ToolResult(
                success=True,
                data={"available_triggers": triggers},
                formatted_text=(
                    "**Available trigger events:**\n"
                    + "\n".join(f"• {t}" for t in triggers)
                    + "\n\nPlease specify a trigger event and optionally a date (YYYY-MM-DD)."
                )
            )

        # Parse the event date
        event_date = self._parse_date(event_date_str)
        if not event_date:
            event_date = datetime.now()
            date_note = " (using today's date — no date was specified)"
        else:
            date_note = ""

        # Find all applicable rules for this trigger
        applicable_rules = {
            key: rule for key, rule in DEADLINE_RULES.items()
            if rule["trigger"] == trigger_event
        }

        if not applicable_rules:
            # If exact trigger not found, show all rules as a reference
            return ToolResult(
                success=True,
                data={"trigger_event": trigger_event},
                formatted_text=self._format_all_rules(trigger_event)
            )

        # Compute deadlines
        deadlines: List[DeadlineResult] = []
        for key, rule in applicable_rules.items():
            due_date = event_date + timedelta(days=rule["days"])
            deadline = DeadlineResult(
                description=rule["description"],
                due_date=due_date.strftime("%Y-%m-%d"),
                days_from_trigger=rule["days"],
                authority=rule["authority"],
                note=rule.get("note", "")
            )
            deadlines.append(deadline)

        # Format output
        lines = [
            f"**Legal Deadlines for '{trigger_event}' from {event_date.strftime('%Y-%m-%d')}{date_note}:**\n"
        ]
        for dl in sorted(deadlines, key=lambda d: d.days_from_trigger):
            lines.append(f"📅 {dl.to_formatted_string()}")
            lines.append("")

        lines.append(
            "\n⚠️ *These are standard CrPC timelines. Actual deadlines may vary "
            "based on court orders, holidays, and case-specific circumstances. "
            "Always verify with the relevant court.*"
        )

        logger.info(
            "Deadline calculated | trigger={} | date={} | deadlines_count={}",
            trigger_event, event_date.strftime("%Y-%m-%d"), len(deadlines)
        )

        return ToolResult(
            success=True,
            data=[{
                "description": dl.description,
                "due_date": dl.due_date,
                "days": dl.days_from_trigger,
                "authority": dl.authority,
                "note": dl.note,
            } for dl in deadlines],
            formatted_text="\n".join(lines)
        )

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string in YYYY-MM-DD or DD/MM/YYYY format.
        
        Args:
            date_str: The date string to parse.
            
        Returns:
            A datetime object, or None if parsing fails.
        """
        if not date_str or not str(date_str).strip():
            return None

        date_str = str(date_str).strip()

        # Try YYYY-MM-DD
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass

        # Try DD/MM/YYYY
        try:
            return datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            pass

        # Try MM/DD/YYYY as fallback
        try:
            return datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            pass

        logger.warning("Could not parse date: '{}'", date_str)
        return None

    @staticmethod
    def _format_all_rules(trigger_event: str) -> str:
        """Format all deadline rules as a reference table."""
        lines = [
            f"No specific rules found for trigger '{trigger_event}'.\n",
            "**All available CrPC deadline rules:**\n"
        ]
        for key, rule in DEADLINE_RULES.items():
            lines.append(
                f"• **{rule['description']}**\n"
                f"  Trigger: {rule['trigger']} | Days: {rule['days']} | "
                f"Authority: {rule['authority']}"
            )
            if rule.get("note"):
                lines.append(f"  Note: {rule['note']}")
            lines.append("")

        return "\n".join(lines)
