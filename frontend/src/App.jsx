  import { useCallback, useMemo, useState, useEffect } from 'react'
import ChatWindow from './components/ChatWindow'
import ClientModal from './components/ClientModal'
import SettingsModal from './components/SettingsModal'
import Header from './components/Header'
import InputBar from './components/InputBar'
import Layout from './components/Layout'
import LoadingScreen from './components/LoadingScreen'
import Sidebar from './components/Sidebar'
import { useChat } from './hooks/useChat'
import { useStreamChat } from './hooks/useStreamChat'

function buildSessionId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

export default function App() {
  const sessionId = useMemo(() => buildSessionId(), [])
  const [draft, setDraft] = useState('')
  const [currentClient, setCurrentClient] = useState(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isClientModalOpen, setIsClientModalOpen] = useState(false)
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false)
  const [clientModalMode, setClientModalMode] = useState('load')

  const [theme, setTheme] = useState(() => {
    // optional: read from localStorage or default dark
    return localStorage.getItem('juris-theme') || 'dark'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('juris-theme', theme)
  }, [theme])

  const handleToggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))
  }, [])

  const {
    messages,
    activeToolName,
    isGenerating,
    queryCount,
    ragHits,
    addUserMessage,
    handleSessionReady,
    handleToken,
    handleDone,
    handleToolInvoked,
    handleToolResult,
    handleRagRetrieved,
    handleError,
    handleConnectionInterrupted,
    isVoiceEnabled,
    setIsVoiceEnabled,
    resetSpeech,
    switchClient,
    loadClientHistory,
  } = useChat()

  const handleToggleVoice = useCallback(() => {
    setIsVoiceEnabled((prev) => {
      const next = !prev
      if (!next) {
        resetSpeech()
      }
      return next
    })
  }, [setIsVoiceEnabled, resetSpeech])

  const handlers = useMemo(
    () => ({
      onSessionReady: handleSessionReady,
      onToken: handleToken,
      onDone: handleDone,
      onToolInvoked: handleToolInvoked,
      onToolResult: handleToolResult,
      onRagRetrieved: handleRagRetrieved,
      onErrorMessage: handleError,
      onClose: handleConnectionInterrupted,
      onSocketError: handleConnectionInterrupted,
      onReconnectExhausted: handleConnectionInterrupted,
    }),
    [
      handleConnectionInterrupted,
      handleDone,
      handleError,
      handleRagRetrieved,
      handleSessionReady,
      handleToken,
      handleToolInvoked,
      handleToolResult,
    ],
  )

  const { status, sendUserMessage } = useStreamChat({
    handlers,
  })

  const handleSend = useCallback(async () => {
    const text = draft.trim()
    if (!text || status !== 'online' || isGenerating) {
      return
    }

    addUserMessage(text)
    setDraft('')

    const isSent = await sendUserMessage({
      sessionId,
      message: text,
      clientId: currentClient?.client_id || null,
    })

    if (!isSent) {
      return
    }

    setIsSidebarOpen(false)
  }, [
    addUserMessage,
    currentClient?.client_id,
    draft,
    isGenerating,
    sendUserMessage,
    sessionId,
    status,
  ])

  const handleOpenClientModal = useCallback((mode) => {
    setClientModalMode(mode)
    setIsClientModalOpen(true)
  }, [])

  const handleClientLoaded = useCallback((client) => {
    setCurrentClient(client)
    switchClient(client?.client_id)
    // Eagerly load and display existing chat history for this client
    if (client?.client_id) {
      loadClientHistory(client.client_id)
    }
  }, [switchClient, loadClientHistory])

  const handleClientCreated = useCallback((client) => {
    setCurrentClient(client)
    switchClient(client?.client_id)
    // New clients have no history, but call for consistency
    if (client?.client_id) {
      loadClientHistory(client.client_id)
    }
  }, [switchClient, loadClientHistory])

  const handleClientClear = useCallback(() => {
    setCurrentClient(null)
    switchClient(null)
  }, [switchClient])

  const handleQuickAction = useCallback((template) => {
    setDraft(template)
    setIsSidebarOpen(false)
  }, [])

  return (
    <>
      <LoadingScreen connected={status !== 'connecting'} />
      <div id="app-root">
        <Layout
          isSidebarOpen={isSidebarOpen}
          onCloseSidebar={() => setIsSidebarOpen(false)}
          header={
            <Header
              status={status}
              onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
              theme={theme}
              onToggleTheme={handleToggleTheme}
            />
          }
          sidebar={
            <Sidebar
              client={currentClient}
              sessionId={sessionId}
              queryCount={queryCount}
              ragHits={ragHits}
              onOpenClientModal={handleOpenClientModal}
              onQuickAction={handleQuickAction}
              onClearClient={handleClientClear}
              onEditClient={() => handleOpenClientModal('create')}
              onOpenSettings={() => setIsSettingsModalOpen(true)}
            />
          }
          main={
            <>
              <ChatWindow
                messages={messages}
                activeToolName={activeToolName}
                isGenerating={isGenerating}
                onQuickAction={handleQuickAction}
              />
              <InputBar
                value={draft}
                onChange={setDraft}
                onSend={handleSend}
                isGenerating={isGenerating}
                disabled={status !== 'online'}
                isVoiceEnabled={isVoiceEnabled}
                onToggleVoice={handleToggleVoice}
              />
            </>
          }
        />
      </div>
      <ClientModal
        open={isClientModalOpen}
        mode={clientModalMode}
        initialClient={currentClient}
        onClose={() => setIsClientModalOpen(false)}
        onClientLoaded={handleClientLoaded}
        onClientCreated={handleClientCreated}
      />
      <SettingsModal 
        open={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        currentClient={currentClient}
        onClientClear={handleClientClear}
      />
    </>
  )
}
