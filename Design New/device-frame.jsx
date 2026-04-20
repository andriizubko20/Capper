// Minimal device frame for Capper — no dependencies, single component
function IOSDevice({ children, width = 402, height = 874, dark = true }) {
  return (
    <div style={{
      width, height, borderRadius: 48, overflow: 'hidden',
      position: 'relative', background: dark ? '#000' : '#F2F2F7',
      boxShadow: '0 40px 80px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.06), 0 0 0 8px #1a1a1d, 0 0 0 9px #2a2a2f',
      fontFamily: '-apple-system, system-ui, sans-serif',
      WebkitFontSmoothing: 'antialiased',
    }}>
      {/* status bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 60,
        zIndex: 20, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '21px 32px 0', boxSizing: 'border-box',
      }}>
        <span style={{
          fontFamily: '-apple-system, "SF Pro", system-ui', fontWeight: 600,
          fontSize: 16, color: '#fff', letterSpacing: 0.2,
        }}>9:41</span>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <svg width="17" height="11" viewBox="0 0 19 12">
            <rect x="0" y="7.5" width="3.2" height="4.5" rx="0.7" fill="#fff"/>
            <rect x="4.8" y="5" width="3.2" height="7" rx="0.7" fill="#fff"/>
            <rect x="9.6" y="2.5" width="3.2" height="9.5" rx="0.7" fill="#fff"/>
            <rect x="14.4" y="0" width="3.2" height="12" rx="0.7" fill="#fff"/>
          </svg>
          <svg width="15" height="11" viewBox="0 0 17 12">
            <path d="M8.5 3.2C10.8 3.2 12.9 4.1 14.4 5.6L15.5 4.5C13.7 2.7 11.2 1.5 8.5 1.5C5.8 1.5 3.3 2.7 1.5 4.5L2.6 5.6C4.1 4.1 6.2 3.2 8.5 3.2Z" fill="#fff"/>
            <path d="M8.5 6.8C9.9 6.8 11.1 7.3 12 8.2L13.1 7.1C11.8 5.9 10.2 5.1 8.5 5.1C6.8 5.1 5.2 5.9 3.9 7.1L5 8.2C5.9 7.3 7.1 6.8 8.5 6.8Z" fill="#fff"/>
            <circle cx="8.5" cy="10.5" r="1.5" fill="#fff"/>
          </svg>
          <svg width="25" height="12" viewBox="0 0 27 13">
            <rect x="0.5" y="0.5" width="23" height="12" rx="3.5" stroke="rgba(255,255,255,0.5)" fill="none"/>
            <rect x="2" y="2" width="20" height="9" rx="2" fill="#fff"/>
            <path d="M25 4.5V8.5C25.8 8.2 26.5 7.2 26.5 6.5C26.5 5.8 25.8 4.8 25 4.5Z" fill="rgba(255,255,255,0.5)"/>
          </svg>
        </div>
      </div>
      {/* dynamic island */}
      <div style={{
        position: 'absolute', top: 11, left: '50%', transform: 'translateX(-50%)',
        width: 126, height: 37, borderRadius: 24, background: '#000', zIndex: 50,
      }} />
      {/* content */}
      {children}
      {/* home indicator */}
      <div style={{
        position: 'absolute', bottom: 8, left: '50%', transform: 'translateX(-50%)',
        zIndex: 60, width: 139, height: 5, borderRadius: 100,
        background: 'rgba(255,255,255,0.55)', pointerEvents: 'none',
      }}/>
    </div>
  );
}

window.IOSDevice = IOSDevice;
