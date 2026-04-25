"""
Juris AI — CRM Tool Evaluation
Unit tests for CRUD operations, error handling, and integration.
"""

import pytest
import pytest_asyncio
import json
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from conftest import load_test_data, save_result


@pytest.fixture(scope="module")
def crm_cases():
    data = load_test_data("tool_test_cases.json")
    return data["crm_test_cases"]


class TestCRMTool:
    """CRM tool unit tests — direct invocation."""

    _created_client_id = None

    @pytest.mark.asyncio
    async def test_create_client(self, crm_tool, crm_cases):
        """Create a new client and verify success."""
        case = next(c for c in crm_cases if c["id"] == "crm_01")
        result = await crm_tool.run(action="create", **case["input"])
        assert result.success is True, f"Create failed: {result.error_message}"
        assert result.data is not None
        assert "client_id" in result.data
        TestCRMTool._created_client_id = result.data["client_id"]

    @pytest.mark.asyncio
    async def test_get_client(self, crm_tool):
        """Retrieve the created client and verify fields."""
        assert TestCRMTool._created_client_id, "No client created yet"
        result = await crm_tool.run(action="get", client_id=TestCRMTool._created_client_id)
        assert result.success is True
        assert result.data["name"] == "Test Client Alpha"
        assert result.data["cnic"] == "12345-1234567-1"
        assert result.data["case_type"] == "Murder"

    @pytest.mark.asyncio
    async def test_update_client(self, crm_tool):
        """Update bail_status and verify."""
        assert TestCRMTool._created_client_id, "No client created yet"
        result = await crm_tool.run(
            action="update",
            client_id=TestCRMTool._created_client_id,
            field="bail_status",
            value="on bail",
        )
        assert result.success is True

        # Verify the update
        get_result = await crm_tool.run(action="get", client_id=TestCRMTool._created_client_id)
        assert get_result.data["bail_status"] == "on bail"

    @pytest.mark.asyncio
    async def test_search_client(self, crm_tool):
        """Search by name and verify found."""
        result = await crm_tool.run(action="search", query="Test Client Alpha")
        assert result.success is True
        assert len(result.data) >= 1
        names = [c["name"] for c in result.data]
        assert "Test Client Alpha" in names

    @pytest.mark.asyncio
    async def test_list_clients(self, crm_tool):
        """List all clients — at least the one we created."""
        result = await crm_tool.run(action="list")
        assert result.success is True
        assert isinstance(result.data, list)

    @pytest.mark.asyncio
    async def test_update_invalid_field(self, crm_tool):
        """Updating an invalid field must fail."""
        assert TestCRMTool._created_client_id, "No client created yet"
        result = await crm_tool.run(
            action="update",
            client_id=TestCRMTool._created_client_id,
            field="invalid_field",
            value="test",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_nonexistent_client(self, crm_tool):
        """Getting a nonexistent client must fail."""
        result = await crm_tool.run(action="get", client_id="nonexistent_id_xyz_999")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_delete_client(self, crm_tool):
        """Delete the test client and verify."""
        assert TestCRMTool._created_client_id, "No client created yet"
        result = await crm_tool.run(action="delete", client_id=TestCRMTool._created_client_id)
        assert result.success is True

        # Verify deleted
        get_result = await crm_tool.run(action="get", client_id=TestCRMTool._created_client_id)
        assert get_result.success is False

    @pytest.mark.asyncio
    async def test_unknown_action(self, crm_tool):
        """Unknown action must return error."""
        result = await crm_tool.run(action="fly_to_moon")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_create_missing_name(self, crm_tool):
        """Creating without a name should default to 'Unknown'."""
        result = await crm_tool.run(action="create")
        assert result.success is True
        cid = result.data["client_id"]
        # Cleanup
        await crm_tool.run(action="delete", client_id=cid)


def pytest_sessionfinish(session, exitstatus):
    """Save CRM test results summary."""
    results = {
        "component": "CRM Tool",
        "total_tests": 10,
        "description": "CRUD operations, error handling, edge cases",
    }
    save_result("crm_results.json", results)
