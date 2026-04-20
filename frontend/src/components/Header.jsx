import StatusDot from './StatusDot'

export default function Header({ status, onToggleSidebar, theme, onToggleTheme }) {
  return (
    <div className="header-inner">
      <div className="header-left">
        <button
          type="button"
          className="menu-button"
          onClick={onToggleSidebar}
          aria-label="Toggle sidebar"
        >
          ☰
        </button>
        <span className="brand-icon">⚖</span>
        <span className="brand-name">Juris AI</span>
      </div>

      <div className="header-tagline">Pakistani Criminal Law Intelligence</div> 
    </div>
  )
}
