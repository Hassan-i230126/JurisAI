import { useCallback, useEffect, useRef, useState } from 'react'

const HEALTH_POLL_MS = 5000

function parseHealth(payload) {
  if (!payload || typeof payload !== 'object') {
    return false
  }

  if (typeof payload.model_loaded === 'boolean') {
    return payload.model_loaded
  }

  return payload.status === 'healthy'
}

export function useStreamChat({ handlers = {} }) {
  const [status, setStatus] = useState('connecting')

  const handlersRef = useRef(handlers)
  const pollRef = useRef(null)
  const activeRequestRef = useRef(null)

  useEffect(() => {
    handlersRef.current = handlers
  }, [handlers])

  const checkServerHealth = useCallback(async () => {
    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => controller.abort(), 2000)

    try {
      const response = await fetch('/api/health', {
        method: 'GET',
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error(`Health endpoint returned ${response.status}`)
      }

      const payload = await response.json()
      const isOnline = parseHealth(payload)
      setStatus(isOnline ? 'online' : 'offline')
      return isOnline
    } catch {
      setStatus('offline')
      return false
    } finally {
      window.clearTimeout(timeoutId)
    }
  }, [])

  useEffect(() => {
    let isMounted = true

    const runInitialCheck = async () => {
      if (!isMounted) {
        return
      }
      await checkServerHealth()
    }

    runInitialCheck()
    pollRef.current = window.setInterval(() => {
      checkServerHealth()
    }, HEALTH_POLL_MS)

    return () => {
      isMounted = false
      if (pollRef.current) {
        window.clearInterval(pollRef.current)
        pollRef.current = null
      }
      if (activeRequestRef.current) {
        activeRequestRef.current.abort()
        activeRequestRef.current = null
      }
    }
  }, [checkServerHealth])

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
      default:
        callbackSet.onUnknown?.(payload)
        break
    }

    callbackSet.onMessage?.(payload)
  }, [])

  const sendUserMessage = useCallback(
    async ({ sessionId, message, clientId = null }) => {
      const isOnline = await checkServerHealth()
      if (!isOnline) {
        handlersRef.current.onErrorMessage?.({
          type: 'error',
          message: 'Server Offline: local Ollama/LLM is unavailable.',
        })
        return false
      }

      if (activeRequestRef.current) {
        activeRequestRef.current.abort()
        activeRequestRef.current = null
      }

      const controller = new AbortController()
      activeRequestRef.current = controller

      try {
        const response = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            session_id: sessionId,
            message,
            client_id: clientId,
          }),
          signal: controller.signal,
        })

        if (!response.ok) {
          const fallbackMessage =
            response.status === 503
              ? 'Server Offline: local Ollama/LLM is unavailable.'
              : `Request failed with status ${response.status}`

          let errorMessage = fallbackMessage
          try {
            const payload = await response.json()
            errorMessage = payload.detail || payload.error || fallbackMessage
          } catch {
            errorMessage = fallbackMessage
          }

          setStatus(response.status === 503 ? 'offline' : 'online')
          handlersRef.current.onErrorMessage?.({ type: 'error', message: errorMessage })
          return false
        }

        const reader = response.body?.getReader()
        if (!reader) {
          handlersRef.current.onErrorMessage?.({
            type: 'error',
            message: 'Streaming not supported in this browser/session.',
          })
          return false
        }

        setStatus('online')
        const decoder = new TextDecoder('utf-8')
        let buffer = ''

        while (true) {
          const { value, done } = await reader.read()
          if (done) {
            break
          }

          buffer += decoder.decode(value, { stream: true })

          let newlineIdx = buffer.indexOf('\n')
          while (newlineIdx !== -1) {
            const line = buffer.slice(0, newlineIdx).trim()
            buffer = buffer.slice(newlineIdx + 1)

            if (line) {
              try {
                const payload = JSON.parse(line)
                dispatchByType(payload)
              } catch {
                handlersRef.current.onParseError?.(line)
              }
            }

            newlineIdx = buffer.indexOf('\n')
          }
        }

        const remainder = buffer.trim()
        if (remainder) {
          try {
            dispatchByType(JSON.parse(remainder))
          } catch {
            handlersRef.current.onParseError?.(remainder)
          }
        }

        handlersRef.current.onClose?.()
        return true
      } catch (error) {
        if (error?.name === 'AbortError') {
          handlersRef.current.onClose?.()
          return false
        }

        setStatus('offline')
        handlersRef.current.onSocketError?.(error)
        handlersRef.current.onErrorMessage?.({
          type: 'error',
          message: 'Network error while streaming response.',
        })
        return false
      } finally {
        if (activeRequestRef.current === controller) {
          activeRequestRef.current = null
        }
      }
    },
    [checkServerHealth, dispatchByType],
  )

  return {
    status,
    sendUserMessage,
  }
}
