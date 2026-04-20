import { useEffect, useRef, useState } from 'react'

export default function CitationBar({ citations }) {
  const [expanded, setExpanded] = useState(false)
  const [maxHeight, setMaxHeight] = useState(0)
  const contentRef = useRef(null)

  useEffect(() => {
    if (!expanded) {
      setMaxHeight(0)
      return
    }

    const node = contentRef.current
    if (!node) {
      return
    }

    setMaxHeight(node.scrollHeight)
  }, [citations, expanded])

  return (
    <div className="citation-bar">
      <button
        type="button"
        className="citation-toggle"
        onClick={() => setExpanded((prev) => !prev)}
      >
        <span>{expanded ? '▾' : '▸'}</span>
        <span>Sources ({citations.length})</span>
      </button>

      <div className="citation-content" style={{ maxHeight: `${maxHeight}px` }}>
        <div className="citation-list" ref={contentRef}>
          {citations.map((citation) => (
            <div className="citation-item" key={citation.id}>
              • {citation.title} [{citation.type}] · relevance: {citation.relevance}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
