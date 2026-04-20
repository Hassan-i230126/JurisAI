export default function Layout({
  header,
  sidebar,
  main,
  isSidebarOpen,
  onCloseSidebar,
}) {
  return (
    <div className="layout">
      <div className="header">{header}</div>
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>{sidebar}</aside>
      <div
        className={`sidebar-backdrop ${isSidebarOpen ? 'open' : ''}`}
        onClick={onCloseSidebar}
        role="presentation"
      />
      <main className="main">{main}</main>
    </div>
  )
}
