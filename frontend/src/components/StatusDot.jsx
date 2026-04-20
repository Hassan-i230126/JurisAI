const LABELS = {
  online: 'Connected',
  connecting: 'Connecting...',
  offline: 'Offline',
}

export default function StatusDot({ status }) {
  const safeStatus = LABELS[status] ? status : 'offline'

  return (
    <div className="status-dot" aria-live="polite">
      <span className={`dot dot-${safeStatus}`} />
      <span>{LABELS[safeStatus]}</span>
    </div>
  )
}
