"""
Juris AI — Tool Functional Tests
Direct invocation tests for statute_lookup, case_search, deadline_calculator.
"""

import pytest
import json
from conftest import load_test_data, save_result


@pytest.fixture(scope="module")
def tool_cases():
    return load_test_data("tool_test_cases.json")


class TestStatuteLookup:
    """Statute lookup tool — direct invocation tests."""

    @pytest.mark.asyncio
    async def test_lookup_section_302(self, statute_tool):
        """Look up Section 302 PPC — murder."""
        result = await statute_tool.run(act="PPC", section_number="302")
        assert result.success is True
        assert "302" in result.formatted_text

    @pytest.mark.asyncio
    async def test_lookup_section_379(self, statute_tool):
        """Look up Section 379 PPC — theft."""
        result = await statute_tool.run(act="PPC", section_number="379")
        assert result.success is True
        assert "379" in result.formatted_text

    @pytest.mark.asyncio
    async def test_lookup_nonexistent_section(self, statute_tool):
        """Non-existent section should fail gracefully."""
        result = await statute_tool.run(act="PPC", section_number="99999")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_keyword_search(self, statute_tool):
        """Keyword search for 'murder' should return results."""
        result = await statute_tool.run(keyword="murder")
        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_missing_args(self, statute_tool):
        """No section_number or keyword should fail."""
        result = await statute_tool.run()
        assert result.success is False


class TestDeadlineCalculator:
    """Deadline calculator tool — direct invocation tests."""

    @pytest.mark.asyncio
    async def test_arrest_deadlines(self, deadline_tool):
        """Calculate deadlines after arrest."""
        result = await deadline_tool.run(trigger_event="arrest", event_date="2024-03-15")
        assert result.success is True
        assert "deadline" in result.formatted_text.lower() or "2024" in result.formatted_text

    @pytest.mark.asyncio
    async def test_conviction_deadlines(self, deadline_tool):
        """Calculate deadlines after conviction."""
        result = await deadline_tool.run(trigger_event="conviction_order", event_date="2024-06-01")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_date_uses_today(self, deadline_tool):
        """No date should default to today."""
        result = await deadline_tool.run(trigger_event="arrest")
        assert result.success is True
        assert "today" in result.formatted_text.lower() or "date" in result.formatted_text.lower()

    @pytest.mark.asyncio
    async def test_no_trigger_shows_all(self, deadline_tool):
        """No trigger event should list all available triggers."""
        result = await deadline_tool.run(trigger_event="")
        assert result.success is True
        assert "trigger" in result.formatted_text.lower() or "available" in result.formatted_text.lower()

    @pytest.mark.asyncio
    async def test_invalid_date_format(self, deadline_tool):
        """Invalid date format should still work (fallback to today)."""
        result = await deadline_tool.run(trigger_event="arrest", event_date="not-a-date")
        assert result.success is True


class TestCaseSearch:
    """Case search tool — direct invocation tests."""

    @pytest.mark.asyncio
    async def test_search_bail_murder(self, case_search_tool):
        """Search for bail in murder cases."""
        result = await case_search_tool.run(query="bail in murder cases")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_empty_query(self, case_search_tool):
        """Empty query should fail."""
        result = await case_search_tool.run(query="")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_search_anti_terrorism(self, case_search_tool):
        """Search for anti-terrorism cases."""
        result = await case_search_tool.run(query="anti-terrorism conviction appeal")
        assert result.success is True


def pytest_sessionfinish(session, exitstatus):
    """Save tool test results."""
    results = {
        "component": "Tools (statute_lookup, case_search, deadline_calculator)",
        "total_tests": 13,
        "description": "Functional correctness, error handling, edge cases",
    }
    save_result("tools_results.json", results)
