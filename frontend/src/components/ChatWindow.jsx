import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import ToolIndicator from './ToolIndicator'

const BOTTOM_THRESHOLD_PX = 120

export default function ChatWindow({ messages, activeToolName, isGenerating, onQuickAction }) {
  const containerRef = useRef(null)
  const endRef = useRef(null)
  const shouldAutoScrollRef = useRef(true)

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return undefined
    }

    const onScroll = () => {
      const distanceFromBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight
      shouldAutoScrollRef.current = distanceFromBottom <= BOTTOM_THRESHOLD_PX
    }

    container.addEventListener('scroll', onScroll, { passive: true })
    onScroll()

    return () => {
      container.removeEventListener('scroll', onScroll)
    }
  }, [])

  useEffect(() => {
    const endNode = endRef.current
    if (!endNode) {
      return
    }

    if (isGenerating || shouldAutoScrollRef.current) {
      endNode.scrollIntoView({ block: 'end', behavior: isGenerating ? 'auto' : 'smooth' })
    }
  }, [activeToolName, isGenerating, messages])

  if (!messages.length) {
    const defaultPrompts = [
      { icon: '⚖️', label: 'Statute Search', text: 'What is the punishment for theft under PPC?' },
      { icon: '🔍', label: 'Case Law', text: 'Find Supreme Court judgments about post-arrest bail' },
      { icon: '📅', label: 'Deadlines', text: 'If bail is refused by Sessions court, when can I apply to High Court?' },
      { icon: '📝', label: 'Procedure', text: 'What is the correct procedure for filing an FIR?' },
    ];

    return (
      <section className="chat-window" ref={containerRef}>
        <div className="chat-window-inner">
          <div className="chat-empty">
            <div className="chat-empty-mark">⚖</div>
            <h2 className="chat-empty-title">Ask Juris AI</h2>
            <p className="chat-empty-body">Your personal Legal AI Companion</p>
            <div className="chat-empty-prompts">
              {defaultPrompts.map((prompt, idx) => (
                <button
                  key={idx}
                  className="chat-prompt-box"
                  onClick={() => onQuickAction && onQuickAction(prompt.text)}
                >
                  <span className="prompt-icon">{prompt.icon}</span>
                  <span className="prompt-label">{prompt.label}</span>
                  <span className="prompt-text">{prompt.text}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="chat-window" ref={containerRef}>
      <div className="chat-window-inner">
        {messages.map((message) => (
          <div
            className={`message-row message-row-${message.role}`}
            key={message.id}
          >
            <MessageBubble message={message} />
          </div>
        ))}

        {activeToolName ? (
          <div className="tool-row">
            <ToolIndicator toolName={activeToolName} />
          </div>
        ) : null}

        <div ref={endRef} aria-hidden="true" />
      </div>
    </section>
  )
}
