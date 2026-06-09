"""
SummaryVault Plugin — Tool Definitions

Registers the vault_submit tool with Hermes tool system.
"""

from hermes_plugin import ToolDefinition, ToolParameter, ToolSchema

VAULT_SUBMIT_TOOL = ToolDefinition(
    schema=ToolSchema(
        name="vault_submit",
        description=(
            "Submit a summary to SummaryVault for encrypted archival. "
            "Use this to permanently save important work results, findings, "
            "reports, or analysis to the local encrypted vault."
        ),
        parameters={
            "title": ToolParameter(
                type="string",
                description="Title for the summary entry",
                required=True,
            ),
            "content": ToolParameter(
                type="string",
                description="The summary content to archive",
                required=True,
            ),
            "tags": ToolParameter(
                type="string",
                description="Comma-separated tags (e.g. 'python,api,design')",
                required=False,
            ),
            "session_id": ToolParameter(
                type="string",
                description="Session identifier (auto-detected if omitted)",
                required=False,
            ),
        },
    ),
    # Handler is registered in __init__.py
    handler=None,
)
