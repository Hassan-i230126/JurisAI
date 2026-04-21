import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import CitationBar from './CitationBar'
import { formatTimestamp, markdownSanitizeSchema } from '../utils/formatters'
import assistantLogo from '../assets/black_white_logo.jpg'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  const cleanContent = message.content.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim()

  return (
    <>
      {!isUser ? (
        <img
          src={assistantLogo}
          alt="Juris AI"
          className="assistant-logo"
          aria-hidden="true"
          loading="lazy"
        />
      ) : null}

      <article className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
        {isUser ? (
          <>
            <div>{cleanContent}</div>

            {!message.isStreaming ? (
              <div className="message-timestamp">{formatTimestamp(message.timestamp)}</div>
            ) : null}
          </>
        ) : (
          <div className="assistant-content">
            <div className="markdown-body">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[[rehypeSanitize, markdownSanitizeSchema]]}
              >
                {cleanContent}
              </ReactMarkdown>
              {message.isStreaming ? <span className="cursor">▌</span> : null}
            </div>

            {!message.isStreaming ? (
              <div className="message-timestamp">{formatTimestamp(message.timestamp)}</div>
            ) : null}

            {!message.isStreaming && message.citations?.length ? (
              <CitationBar citations={message.citations.slice(0, 3)} />
            ) : null}
          </div>
        )}
      </article>

      {isUser ? (
        <div className="user-logo">👤</div>
      ) : null}
    </>
  )
}
