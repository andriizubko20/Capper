import type { Screen } from '@/lib/types'

const ITEMS = [
  {
    id: 'picks' as Screen,
    label: 'Picks',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="5" width="18" height="16" rx="3" stroke="currentColor" strokeWidth="1.8"/>
        <path d="M3 9h18M8 3v4M16 3v4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    id: 'stats' as Screen,
    label: 'Statistics',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <path d="M4 19V8M10 19V4M16 19v-7M22 19H2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    id: 'compare' as Screen,
    label: 'Compare',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <path d="M8 3v18M16 3v18M3 8h5M3 16h5M16 8h5M16 16h5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
      </svg>
    ),
  },
]

interface Props {
  active: Screen
  onChange: (s: Screen) => void
}

export function BottomNav({ active, onChange }: Props) {
  return (
    <div className="bottom-nav">
      {ITEMS.map(item => (
        <div
          key={item.id}
          className={`nav-item${active === item.id ? ' active' : ''}`}
          onClick={() => onChange(item.id)}
        >
          {item.icon}
          <div className="nav-label">{item.label}</div>
        </div>
      ))}
    </div>
  )
}
