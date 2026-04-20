import { useEffect, useState } from 'react'

export default function LoadingScreen({ connected }) {
  const [mounted, setMounted] = useState(true)
  const [fading, setFading] = useState(false)

  useEffect(() => {
    if (!connected || !mounted) {
      return
    }

    setFading(true)
    const timer = window.setTimeout(() => {
      setMounted(false)
    }, 500)

    return () => window.clearTimeout(timer)
  }, [connected, mounted])

  if (!mounted) {
    return null
  }

  return (
    <div className={`loading-screen ${fading ? 'fade-out' : ''}`}>
      <div className="loading-content">
        <h1 className="loading-title">Juris AI</h1>
        <p className="loading-tagline">Pakistani Criminal Law Intelligence</p>
        <p className="loading-state">Initializing...</p>
        <div className="loading-line" />
      </div>
    </div>
  )
}
