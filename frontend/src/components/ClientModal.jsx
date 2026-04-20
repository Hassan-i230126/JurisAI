import { useEffect, useMemo, useState } from 'react'

const DEFAULT_CLIENT = {
  client_id: '',
  name: '',
  case_type: '',
  charges: '',
  bail_status: 'unknown',
  court_name: '',
  next_hearing_date: '',
  notes: '',
}

function buildClientDraft(initialClient) {
  if (!initialClient) {
    return DEFAULT_CLIENT
  }

  return {
    client_id: initialClient.client_id || '',
    name: initialClient.name || '',
    case_type: initialClient.case_type || '',
    charges: initialClient.charges || '',
    bail_status: initialClient.bail_status || 'unknown',
    court_name: initialClient.court_name || '',
    next_hearing_date: initialClient.next_hearing_date || '',
    notes: initialClient.notes || '',
  }
}

export default function ClientModal({
  open,
  mode,
  initialClient,
  onClose,
  onClientLoaded,
  onClientCreated,
}) {
  const [activeMode, setActiveMode] = useState(mode || 'load')
  const [loadClientId, setLoadClientId] = useState('')
  const [clientDraft, setClientDraft] = useState(DEFAULT_CLIENT)
  const [errorText, setErrorText] = useState('')
  const [isBusy, setIsBusy] = useState(false)

  useEffect(() => {
    if (!open) {
      return
    }

    setActiveMode(mode || 'load')
    setLoadClientId(initialClient?.client_id || '')
    setClientDraft(buildClientDraft(initialClient))
    setErrorText('')
  }, [initialClient, mode, open])

  useEffect(() => {
    if (!open) {
      return
    }

    const onEscape = (event) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', onEscape)
    return () => window.removeEventListener('keydown', onEscape)
  }, [onClose, open])

  const isLoadMode = activeMode === 'load'
  const title = useMemo(() => (isLoadMode ? 'Load Client' : 'Client Profile'), [isLoadMode])

  if (!open) {
    return null
  }

  const handleLoadClient = async () => {
    const candidate = loadClientId.trim()
    if (!candidate) {
      setErrorText('Client ID is required.')
      return
    }

    setIsBusy(true)
    setErrorText('')

    try {
      const response = await fetch(`/api/clients/${encodeURIComponent(candidate)}`)
      const data = await response.json()

      if (!response.ok || !data.client) {
        throw new Error(data.detail || 'Client not found.')
      }

      onClientLoaded(data.client)
      onClose()
    } catch (error) {
      setErrorText(error.message || 'Unable to load client.')
    } finally {
      setIsBusy(false)
    }
  }

  const handleCreateClient = async () => {
    if (!clientDraft.client_id.trim()) {
      setErrorText('Client ID is required.')
      return
    }
    if (!clientDraft.name.trim()) {
      setErrorText('Client name is required.')
      return
    }

    setIsBusy(true)
    setErrorText('')

    try {
      const payload = {
        ...clientDraft,
        client_id: clientDraft.client_id.trim(),
        name: clientDraft.name.trim(),
      }

      const response = await fetch('/api/clients', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })
      const data = await response.json()

      if (!response.ok || !data.data) {
        throw new Error(data.detail || 'Client creation failed.')
      }

      onClientCreated({
        ...payload,
        ...data.data,
      })
      onClose()
    } catch (error) {
      setErrorText(error.message || 'Unable to create client.')
    } finally {
      setIsBusy(false)
    }
  }

  return (
    <div className="client-modal-overlay" onClick={onClose} role="presentation">
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <button type="button" className="modal-close" onClick={onClose}>
          ✕
        </button>

        <h2 className="modal-title">{title}</h2>

        <div className="modal-switch">
          <button
            type="button"
            className={isLoadMode ? 'active' : ''}
            onClick={() => setActiveMode('load')}
          >
            Load Existing
          </button>
          <button
            type="button"
            className={!isLoadMode ? 'active' : ''}
            onClick={() => setActiveMode('create')}
          >
            + New Client
          </button>
        </div>

        {isLoadMode ? (
          <div className="form-grid">
            <label>
              <span className="form-label">Client ID</span>
              <input
                className="form-input"
                value={loadClientId}
                onChange={(event) => setLoadClientId(event.target.value)}
                placeholder="Enter client UUID"
              />
            </label>
          </div>
        ) : (
          <div className="form-grid" style={{ maxHeight: '60vh', overflowY: 'auto', paddingRight: '0.5rem' }}>
            <label>
              <span className="form-label">Client ID *</span>
              <input
                className="form-input"
                value={clientDraft.client_id}
                onChange={(event) =>
                  setClientDraft((prev) => ({ ...prev, client_id: event.target.value }))
                }
                placeholder="Unique identifier"
              />
            </label>

            <label>
              <span className="form-label">Name *</span>
              <input
                className="form-input"
                value={clientDraft.name}
                onChange={(event) =>
                  setClientDraft((prev) => ({ ...prev, name: event.target.value }))
                }
                placeholder="Full name"
              />
            </label>

            <label>
              <span className="form-label">Case Type</span>
              <select
                className="form-input"
                value={clientDraft.case_type}
                onChange={(event) =>
                  setClientDraft((prev) => ({ ...prev, case_type: event.target.value }))
                }
              >
                <option value="">Select Case Type...</option>
                <option value="Criminal">Criminal</option>
                <option value="Civil">Civil</option>
                <option value="Corporate">Corporate</option>
                <option value="Family">Family</option>
                <option value="Constitutional">Constitutional</option>
                <option value="Property">Property</option>
                <option value="Other">Other</option>
              </select>
            </label>

            <label>
              <span className="form-label">Charges</span>
              <input
                className="form-input"
                value={clientDraft.charges}
                onChange={(event) =>
                  setClientDraft((prev) => ({ ...prev, charges: event.target.value }))
                }
                placeholder="PPC 302, 324"
              />
            </label>

            <label>
              <span className="form-label">Bail Status</span>
              <select
                className="form-input"
                value={clientDraft.bail_status}
                onChange={(event) =>
                  setClientDraft((prev) => ({ ...prev, bail_status: event.target.value }))
                }
              >
                <option value="unknown">Unknown</option>
                <option value="Not Applied">Not Applied</option>
                <option value="Pending">Pending</option>
                <option value="Granted">Granted</option>
                <option value="Rejected">Rejected</option>
                <option value="Cancelled">Cancelled</option>
              </select>
            </label>

            <label>
              <span className="form-label">Court</span>
              <input
                className="form-input"
                value={clientDraft.court_name}
                onChange={(event) =>
                  setClientDraft((prev) => ({ ...prev, court_name: event.target.value }))
                }
                placeholder="Court name"
              />
            </label>

            <label>
              <span className="form-label">Next Hearing</span>
              <input
                className="form-input"
                type="date"
                value={clientDraft.next_hearing_date}
                onChange={(event) =>
                  setClientDraft((prev) => ({
                    ...prev,
                    next_hearing_date: event.target.value,
                  }))
                }
              />
            </label>

            <label>
              <span className="form-label">Notes</span>
              <textarea
                className="form-input"
                value={clientDraft.notes}
                onChange={(event) =>
                  setClientDraft((prev) => ({ ...prev, notes: event.target.value }))
                }
                rows={3}
                placeholder="Additional context"
              />
            </label>
          </div>
        )}

        {errorText ? <p className="modal-error">{errorText}</p> : null}

        <div className="modal-actions">
          <button type="button" className="modal-button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="modal-button primary"
            onClick={isLoadMode ? handleLoadClient : handleCreateClient}
            disabled={isBusy}
          >
            {isBusy ? 'Please wait...' : isLoadMode ? 'Load Client' : 'Create Client'}
          </button>
        </div>
      </div>
    </div>
  )
}
