function getBailPillClass(status) {
  const value = (status || '').toLowerCase()
  if (value.includes('custody')) {
    return 'bail-pill-custody'
  }
  if (value.includes('bail') || value.includes('acquitted')) {
    return 'bail-pill-bail'
  }
  return 'bail-pill-pending'
}

export default function Sidebar({
  client,
  sessionId,
  queryCount,
  ragHits,
  onOpenClientModal,
  onQuickAction,
  onClearClient,
  onEditClient,
  onOpenSettings,
}) {
  return (
    <div className="sidebar-content">
      <section className="sidebar-section">
        <h2 className="section-title">Current Client</h2>
        <div className="client-card">
          {!client ? (
            <>
              <p className="client-empty">No client loaded</p>
              <div className="client-actions">
                <button
                  type="button"
                  className="inline-button"
                  onClick={() => onOpenClientModal('load')}
                >
                  Load Client
                </button>
                <button
                  type="button"
                  className="inline-button"
                  onClick={() => onOpenClientModal('create')}
                >
                  + New
                </button>
              </div>
            </>
          ) : (
            <>
              <p className="client-name">● {client.name || 'Unnamed Client'}</p>
              <p className="client-meta client-charges">
                Charges: {client.charges || 'N/A'}
              </p>
              <p className="client-meta">
                Bail:
                <span className={`bail-pill ${getBailPillClass(client.bail_status)}`}>
                  {client.bail_status || 'Unknown'}
                </span>
              </p>
              <p className="client-meta">
                Next Hearing: {client.next_hearing_date || 'Not set'}
              </p>

              <div className="client-actions">
                <button type="button" className="inline-button" onClick={onEditClient}>
                  Edit
                </button>
                <button
                  type="button"
                  className="inline-button"
                  onClick={onClearClient}
                >
                  Clear
                </button>
              </div>
            </>
          )}
        </div>
      </section>

      <section className="sidebar-section">
        <h2 className="section-title">Quick Actions</h2>
        <button
          type="button"
          className="sidebar-quick-button"
          onClick={() => onQuickAction('Look up Section ___ of ___.')}
        >
          ⚖ Statute Lookup
        </button>
        <button
          type="button"
          className="sidebar-quick-button"
          onClick={() =>
            onQuickAction('Calculate legal deadline for ___ under CrPC provisions.')
          }
        >
          📅 Deadline Calculator
        </button>
        <button
          type="button"
          className="sidebar-quick-button"
          onClick={() => onQuickAction('Search criminal case law regarding ___.')}
        >
          🔍 Case Search
        </button>
      </section>

      <section className="sidebar-bottom-bar">
        <div className="sidebar-user-profile">
          <svg className="user-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
          <span>User Profile</span>
        </div>
        <button
          className="sidebar-settings-btn"
          title="Settings"
          onClick={onOpenSettings}
        >
          <svg className="settings-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
      </section>
    </div>
  )
}
