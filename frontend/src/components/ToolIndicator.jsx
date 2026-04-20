const TOOL_TEXT = {
  statute_lookup: 'Looking up Section 302 PPC...',
  deadline_calculator: 'Calculating legal deadline...',
  case_search: 'Searching relevant judgments...',
  crm_tool: 'Retrieving client profile...',
}

export default function ToolIndicator({ toolName }) {
  const text = TOOL_TEXT[toolName] || `Running ${toolName}...`

  return <div className="tool-indicator">⚖ {text}</div>
}
