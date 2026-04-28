import { useEffect, useState } from 'react'
import { apiUrl } from '../utils/config'

export default function SettingsModal({ open, onClose, currentClient, onClientClear }) {
  const [clients, setClients] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!open) return
    loadClients()
  }, [open])

  const loadClients = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(apiUrl('/api/clients'))
      if (!res.ok) throw new Error('Failed to fetch clients')
      const data = await res.json()
      setClients(data.clients || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteClient = async (clientId) => {
    if (!window.confirm('Are you sure you want to delete this client? This action cannot be undone.')) {
      return
    }

    try {
      const res = await fetch(apiUrl(`/api/clients/${clientId}`), {
        method: 'DELETE',
      })
      
      if (!res.ok) {
        const errorData = await res.json()
        throw new Error(errorData.detail || 'Failed to delete client')
      }
      
      // Update local state
      setClients((prev) => prev.filter((c) => c.client_id !== clientId))

      // If the active client was deleted, clear it from view
      if (currentClient && currentClient.client_id === clientId) {
        if (onClientClear) onClientClear()
      }

    } catch (err) {
      alert(`Error deleting client: ${err.message}`)
    }
  }

  if (!open) return null

  return (
    <div className="client-modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">Settings</h2>
          <button type="button" className="inline-button" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body" style={{ marginTop: '16px' }}>
          <h3>Client Profiles</h3>
          {loading ? (
            <p>Loading clients...</p>
          ) : error ? (
            <p className="error-text">{error}</p>
          ) : clients.length === 0 ? (
            <p>No clients available.</p>
          ) : (
            <div className="settings-clients-list">
              {clients.map((client) => (
                <div key={client.client_id} className="settings-client-card">
                  <div className="settings-client-info">
                    <p className="settings-client-name">{client.name || 'Unnamed Client'}</p>
                    <p className="settings-client-id">ID: {client.client_id}</p>
                  </div>
                  <button 
                    className="settings-delete-btn" 
                    title="Delete Client"
                    onClick={() => handleDeleteClient(client.client_id)}
                  >
                    <svg className="dustbin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}