import { useCallback, useEffect, useRef, useState } from 'react'

export default function InputBar({
  value,
  onChange,
  onSend,
  isGenerating,
  disabled,
  isVoiceEnabled,
  onToggleVoice,
}) {
  const textareaRef = useRef(null)
  const [isListening, setIsListening] = useState(false)
  const recognitionRef = useRef(null)

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = false;
        recognitionRef.current.interimResults = false;

        recognitionRef.current.onresult = (event) => {
          const transcript = event.results[event.results.length - 1][0].transcript;
          onChange(value ? `${value} ${transcript}` : transcript);
        };

        recognitionRef.current.onerror = () => {
          setIsListening(false);
        };

        recognitionRef.current.onend = () => {
          setIsListening(false);
        };
      }
    }
  }, [onChange, value]);

  const toggleListening = useCallback(() => {
    if (!recognitionRef.current) {
      alert('Speech recognition is not supported in your browser.');
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
    }
  }, [isListening]);

  const resizeTextarea = useCallback(() => {
    const node = textareaRef.current
    if (!node) {
      return
    }

    node.style.height = 'auto'
    node.style.height = `${Math.min(node.scrollHeight, 120)}px`
  }, [])

  useEffect(() => {
    resizeTextarea()
  }, [resizeTextarea, value])

  const handleKeyDown = useCallback(
    (event) => {
      if (event.key !== 'Enter' || event.shiftKey) {
        return
      }

      event.preventDefault()
      onSend()
    },
    [onSend],
  )

  const handleSend = useCallback(() => {
    onSend()
    const node = textareaRef.current
    if (node) {
      node.style.height = '44px'
    }
  }, [onSend])

  const isBlocked = disabled || isGenerating || !value.trim()

  return (
    <section className="input-bar">
      <div className="input-row">
        <textarea
          ref={textareaRef}
          className="input-textarea"
          placeholder="Ask about Pakistani criminal law..."
          value={value}
          disabled={disabled || isGenerating}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <button
            type="button"
            className="icon-button"
            onClick={onToggleVoice}
            title={isVoiceEnabled ? 'Disable Narration' : 'Enable Narration'}
            aria-label="Toggle narration"
            style={{ 
              opacity: isVoiceEnabled ? 1 : 0.5,
              borderRadius: '50%',
              width: '36px',
              height: '36px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: 'none',
              background: 'white',
              cursor: 'pointer',
              fontSize: '1.2rem'
            }}
          >
            {isVoiceEnabled ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <line x1="23" y1="9" x2="17" y2="15"></line>
                <line x1="17" y1="9" x2="23" y2="15"></line>
              </svg>
            )}
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={toggleListening}
            title={isListening ? 'Stop Listening' : 'Start Voice Input'}
            aria-label="Voice Input"
            style={{ 
              color: isListening ? '#ef4444' : 'inherit',
              borderRadius: '50%',
              width: '36px',
              height: '36px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: 'none',
              background: 'none',
              cursor: 'pointer',
              fontSize: '1.2rem'
            }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <line x1="12" y1="19" x2="12" y2="22"></line>
            </svg>
          </button>
          <button
            type="button"
            className="send-button"
            onClick={handleSend}
            disabled={isBlocked}
            aria-label="Send message"
            style={{
              borderRadius: '50%',
              width: '36px',
              height: '36px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 0
            }}
          >
            {isGenerating ? <span className="spinner" /> : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="19" x2="12" y2="5"></line>
                <polyline points="5 12 12 5 19 12"></polyline>
              </svg>
            )}
          </button>
        </div>
      </div>
    </section>
  )
}
