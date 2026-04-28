import { useCallback, useEffect, useRef, useState } from 'react'
import { formatCitations } from '../utils/formatters'
import { apiUrl } from '../utils/config'

const SPEECH_MIN_CHARS = 60
const SPEECH_MAX_CHARS = 180

function buildId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

function createMessage(role, content, overrides = {}) {
  return {
    id: buildId(),
    role,
    content,
    isStreaming: false,
    citations: [],
    toolsUsed: [],
    timestamp: new Date(),
    ...overrides,
  }
}

export function useChat() {
  const [messagesDict, setMessagesDict] = useState({ default: [] })
  const [currentClientId, setCurrentClientId] = useState('default')
  const clientIdRef = useRef('default')

  const [activeToolName, setActiveToolName] = useState(null)
  
  const messages = messagesDict[currentClientId] || []

  const switchClient = useCallback((clientId) => {
    const id = clientId || 'default'
    setCurrentClientId(id)
    clientIdRef.current = id
  }, [])

  /**
   * Eagerly load a client's saved chat history from the backend
   * and populate the message list without requiring the user to
   * send a prompt first.
   */
  const loadClientHistory = useCallback(async (clientId) => {
    if (!clientId) return
    try {
      const res = await fetch(apiUrl(`/api/clients/${encodeURIComponent(clientId)}/history`))
      if (!res.ok) return
      const data = await res.json()
      if (!Array.isArray(data.history) || !data.history.length) return
      const formatted = data.history.map((msg) =>
        createMessage(msg.role, msg.content, {
          timestamp: new Date(msg.timestamp || Date.now()),
          citations: msg.metadata?.citations || [],
          toolsUsed: msg.metadata?.tools_used || [],
        })
      )
      // Replace the slate for this client with the loaded history.
      // Use the functional updater so we always operate on latest state.
      setMessagesDict((dict) => ({
        ...dict,
        [clientId]: formatted,
      }))
    } catch {
      // Non-critical — silently ignore network/parse errors
    }
  }, [])

  const [isGenerating, setIsGenerating] = useState(false)
  const [queryCount, setQueryCount] = useState(0)
  const [ragHits, setRagHits] = useState(0)
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true)

  const streamingMessageIdRef = useRef(null)
  const pendingCitationsRef = useRef([])
  const turnOpenRef = useRef(false)
  const speechQueueRef = useRef([])
  const speechBufferRef = useRef('')
  const speechSpeakingRef = useRef(false)
  const speechVoiceRef = useRef(null)

  const supportsSpeech = useCallback(() => {
    return typeof window !== 'undefined' && 'speechSynthesis' in window
  }, [])

  const selectVoice = useCallback(() => {
    if (!supportsSpeech()) {
      return null
    }

    if (speechVoiceRef.current) {
      return speechVoiceRef.current
    }

    const voices = window.speechSynthesis.getVoices()
    const preferred =
      voices.find((voice) => voice.lang?.toLowerCase().startsWith('en-pk')) ||
      voices.find((voice) => voice.lang?.toLowerCase().startsWith('en')) ||
      voices[0] ||
      null

    speechVoiceRef.current = preferred
    return preferred
  }, [supportsSpeech])

  const speakNextChunk = useCallback(() => {
    if (!supportsSpeech()) {
      return
    }

    if (speechSpeakingRef.current) {
      return
    }

    const chunk = speechQueueRef.current.shift()
    if (!chunk) {
      return
    }

    const utterance = new SpeechSynthesisUtterance(chunk)
    const selectedVoice = selectVoice()
    if (selectedVoice) {
      utterance.voice = selectedVoice
    }
    utterance.rate = 1.0
    utterance.pitch = 1.0

    speechSpeakingRef.current = true
    utterance.onend = () => {
      speechSpeakingRef.current = false
      speakNextChunk()
    }
    utterance.onerror = () => {
      speechSpeakingRef.current = false
      speakNextChunk()
    }

    window.speechSynthesis.speak(utterance)
  }, [selectVoice, supportsSpeech])

  const enqueueSpeechChunk = useCallback(
    (text) => {
      if (!isVoiceEnabled) {
        return
      }
      const clean = text.trim()
      if (!clean || !supportsSpeech()) {
        return
      }

      speechQueueRef.current.push(clean)
      speakNextChunk()
    },
    [speakNextChunk, supportsSpeech, isVoiceEnabled]
  )

  const flushSpeechBuffer = useCallback(() => {
    const pending = speechBufferRef.current.trim()
    if (!pending) {
      return
    }

    speechBufferRef.current = ''
    enqueueSpeechChunk(pending)
  }, [enqueueSpeechChunk])

  const maybeQueueSpeechChunk = useCallback(
    (tokenText) => {
      if (!supportsSpeech() || !tokenText) {
        return
      }

      speechBufferRef.current += tokenText
      const current = speechBufferRef.current
      if (current.length < SPEECH_MIN_CHARS) {
        return
      }

      const shouldFlush = /[.!?\n]\s*$/.test(current) || current.length >= SPEECH_MAX_CHARS
      if (!shouldFlush) {
        return
      }

      speechBufferRef.current = ''
      enqueueSpeechChunk(current)
    },
    [enqueueSpeechChunk, supportsSpeech],
  )

  const resetSpeech = useCallback(() => {
    speechQueueRef.current = []
    speechBufferRef.current = ''
    speechSpeakingRef.current = false

    if (!supportsSpeech()) {
      return
    }

    window.speechSynthesis.cancel()
  }, [supportsSpeech])

  const setMessages = useCallback(
    (updater) => {
      setMessagesDict((dict) => {
        const activeId = clientIdRef.current
        const prev = dict[activeId] || []
        const next = typeof updater === 'function' ? updater(prev) : updater
        return { ...dict, [activeId]: next }
      })
    },
    []
  )

  const addUserMessage = useCallback((content) => {
    resetSpeech()
    setMessages((prev) => [...prev, createMessage('user', content)])
    setIsGenerating(true)
    setQueryCount((prev) => prev + 1)
    turnOpenRef.current = true
    streamingMessageIdRef.current = null
    pendingCitationsRef.current = []
  }, [resetSpeech])

  const handleSessionReady = useCallback((payload) => {
    if (!turnOpenRef.current && !streamingMessageIdRef.current) {
      setIsGenerating(false)
    }
    if (payload && Array.isArray(payload.history) && payload.history.length) {
      const formattedHistory = payload.history.map((msg) =>
        createMessage(msg.role, msg.content, {
          timestamp: new Date(msg.timestamp || Date.now()),
          citations: msg.metadata?.citations || [],
          toolsUsed: msg.metadata?.tools_used || [],
        })
      )
      setMessages((prev) => {
        // The server history contains saved messages. Any user message in
        // the local state whose exact content is NOT already represented in
        // the history is a "pending" bubble — typically the message the user
        // just sent before session_ready arrived. Append those after the
        // history so they stay visible while the assistant streams its reply.
        const historyContents = new Set(
          formattedHistory.filter((m) => m.role === 'user').map((m) => m.content)
        )
        const pendingMessages = prev.filter(
          (m) => m.role === 'user' && !m.isStreaming && !historyContents.has(m.content)
        )
        return pendingMessages.length
          ? [...formattedHistory, ...pendingMessages]
          : formattedHistory
      })
    }
  }, [setMessages])

  const handleToken = useCallback((payload) => {
    if (!turnOpenRef.current) {
      return
    }

    const tokenText = payload.content || ''
    maybeQueueSpeechChunk(tokenText)
    setIsGenerating(true)

    setMessages((prev) => {
      const latestStreaming = [...prev]
        .reverse()
        .find((entry) => entry.role === 'assistant' && entry.isStreaming)

      const activeId = streamingMessageIdRef.current || latestStreaming?.id || null

      if (!activeId) {
        const initialAssistant = createMessage('assistant', tokenText, {
          isStreaming: true,
        })
        streamingMessageIdRef.current = initialAssistant.id
        return [...prev, initialAssistant]
      }

      let found = false
      const next = prev.map((entry) => {
        if (entry.id !== activeId) {
          return entry
        }

        found = true
        return {
          ...entry,
          content: `${entry.content}${tokenText}`,
          isStreaming: true,
        }
      })

      if (!found) {
        const fallbackAssistant = createMessage('assistant', tokenText, {
          isStreaming: true,
        })
        streamingMessageIdRef.current = fallbackAssistant.id
        return [...next, fallbackAssistant]
      }

      streamingMessageIdRef.current = activeId
      return next
    })
  }, [maybeQueueSpeechChunk])

  const handleRagRetrieved = useCallback((payload) => {
    pendingCitationsRef.current = formatCitations(payload.citations || [])
  }, [])

  const handleToolInvoked = useCallback((payload) => {
    setActiveToolName(payload.tool_name || null)
  }, [])

  const handleToolResult = useCallback(() => {
    setActiveToolName(null)
  }, [])

  const handleDone = useCallback((payload) => {
    flushSpeechBuffer()
    const streamingId = streamingMessageIdRef.current
    const citations = formatCitations(
      payload.citations?.length ? payload.citations : pendingCitationsRef.current,
    )

    setMessages((prev) => {
      const fallbackStreaming = [...prev]
        .reverse()
        .find((entry) => entry.role === 'assistant' && entry.isStreaming)
      const targetId = streamingId || fallbackStreaming?.id || null

      if (!targetId) {
        return prev
      }

      return prev.map((entry) => {
        if (entry.id !== targetId) {
          return entry.isStreaming
            ? {
                ...entry,
                isStreaming: false,
              }
            : entry
        }

        return {
          ...entry,
          isStreaming: false,
          citations,
          toolsUsed: payload.tools_used || [],
        }
      })
    })

    if (payload.rag_used) {
      setRagHits((prev) => prev + 1)
    }

    setActiveToolName(null)
    setIsGenerating(false)
    turnOpenRef.current = false
    streamingMessageIdRef.current = null
    pendingCitationsRef.current = []
  }, [flushSpeechBuffer])

  const handleError = useCallback((payload) => {
    const message = payload?.message || 'An unexpected error occurred.'
    resetSpeech()

    setMessages((prev) => [
      ...prev,
      createMessage('assistant', `Warning: ${message}`),
    ])

    setActiveToolName(null)
    setIsGenerating(false)
    turnOpenRef.current = false
    streamingMessageIdRef.current = null
    pendingCitationsRef.current = []
  }, [resetSpeech])

  const handleConnectionInterrupted = useCallback((isError = false) => {
    if (isError) {
      resetSpeech()
    }
    const streamingId = streamingMessageIdRef.current

    if (streamingId) {
      setMessages((prev) =>
        prev.map((entry) => {
          if (entry.id !== streamingId) {
            return entry
          }

          return {
            ...entry,
            isStreaming: false,
          }
        }),
      )
    }

    setActiveToolName(null)
    setIsGenerating(false)
    turnOpenRef.current = false
    streamingMessageIdRef.current = null
    pendingCitationsRef.current = []
  }, [resetSpeech])

  useEffect(() => {
    return () => {
      resetSpeech()
    }
  }, [resetSpeech])

  return {
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
  }
}
