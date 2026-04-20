import { useCallback, useEffect, useRef, useState } from 'react'

const MAX_RECONNECT_ATTEMPTS = 10
const RECONNECT_DELAY_MS = 5000
const HEARTBEAT_INTERVAL_MS = 30000

export function useWebSocket({ sessionId, handlers = {} }) {
  const [status, setStatus] = useState('connecting')
  const [lastMessage, setLastMessage] = useState(null)

  const wsRef = useRef(null)
  const handlersRef = useRef(handlers)
  const reconnectTimerRef = useRef(null)
  const heartbeatRef = useRef(null)
  const reconnectAttemptsRef = useRef(0)
  const shouldReconnectRef = useRef(true)

  useEffect(() => {
    handlersRef.current = handlers
  }, [handlers])

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }, [])

  const stopHeartbeat = useCallback(() => {
    if (heartbeatRef.current) {
      window.clearInterval(heartbeatRef.current)
      heartbeatRef.current = null
    }
  }, [])

  const startHeartbeat = useCallback(() => {
    stopHeartbeat()
    heartbeatRef.current = window.setInterval(() => {
      const socket = wsRef.current
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'ping' }))
      }
    }, HEARTBEAT_INTERVAL_MS)
  }, [stopHeartbeat])

  const dispatchByType = useCallback((payload) => {
    const callbackSet = handlersRef.current

    switch (payload.type) {
      case 'session_ready':
        callbackSet.onSessionReady?.(payload)
        break
      case 'token':
        callbackSet.onToken?.(payload)
        break
      case 'done':
        callbackSet.onDone?.(payload)
        break
      case 'tool_invoked':
        callbackSet.onToolInvoked?.(payload)
        break
      case 'tool_result':
        callbackSet.onToolResult?.(payload)
        break
      case 'rag_retrieved':
        callbackSet.onRagRetrieved?.(payload)
        break
      case 'error':
        callbackSet.onErrorMessage?.(payload)
        break
      case 'pong':
        callbackSet.onPong?.(payload)
        break
      default:
        callbackSet.onUnknown?.(payload)
        break
    }

    callbackSet.onMessage?.(payload)
  }, [])

  const connect = useCallback(() => {
    clearReconnectTimer()
    setStatus('connecting')

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${protocol}://${window.location.host}/ws/${sessionId}`
    const socket = new WebSocket(url)
    wsRef.current = socket

    socket.onopen = () => {
      reconnectAttemptsRef.current = 0
      setStatus('online')
      startHeartbeat()
      handlersRef.current.onOpen?.(socket)
    }

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        setLastMessage(payload)
        dispatchByType(payload)
      } catch {
        handlersRef.current.onParseError?.(event.data)
      }
    }

    socket.onerror = (event) => {
      handlersRef.current.onSocketError?.(event)
    }

    socket.onclose = (event) => {
      stopHeartbeat()
      setStatus('offline')
      handlersRef.current.onClose?.(event)

      if (!shouldReconnectRef.current) {
        return
      }

      if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
        handlersRef.current.onReconnectExhausted?.()
        return
      }

      reconnectAttemptsRef.current += 1
      reconnectTimerRef.current = window.setTimeout(() => {
        connect()
      }, RECONNECT_DELAY_MS)
    }
  }, [clearReconnectTimer, dispatchByType, sessionId, startHeartbeat, stopHeartbeat])

  useEffect(() => {
    shouldReconnectRef.current = true
    connect()

    return () => {
      shouldReconnectRef.current = false
      clearReconnectTimer()
      stopHeartbeat()

      const socket = wsRef.current
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        socket.close()
      }
    }
  }, [clearReconnectTimer, connect, stopHeartbeat])

  const sendMessage = useCallback((payload) => {
    const socket = wsRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false
    }

    socket.send(JSON.stringify(payload))
    return true
  }, [])

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0
    clearReconnectTimer()

    const socket = wsRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.close()
      return
    }

    connect()
  }, [clearReconnectTimer, connect])

  return {
    status,
    sendMessage,
    lastMessage,
    reconnect,
  }
}
