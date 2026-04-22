import { useEffect, useState, useRef } from 'react'
import { getHistory, type HistoryDay } from '@/lib/api'
import type { Pick, Model } from '@/lib/types'

const DOW_UK = ['нд', 'пн', 'вт', 'ср', 'чт', 'пт', 'сб']
const MONTHS_UK = [
  'січня','лютого','березня','квітня','травня','червня',
  'липня','серпня','вересня','жовтня','листопада','грудня',
]

function formatDate(iso: string) {
  // Parse as UTC noon — "2026-04-22" becomes UTC midnight, getDay()/getDate() would
  // return the wrong value for UTC− timezones; UTC methods with noon anchor are safe
  const d = new Date(iso + 'T12:00:00Z')
  return `${DOW_UK[d.getUTCDay()]}, ${d.getUTCDate()} ${MONTHS_UK[d.getUTCMonth()]}`
}

function pnlOf(day: HistoryDay) {
  return day.picks.reduce((s, p) => s + (p.pnl ?? 0), 0)
}

// ── Compact row ───────────────────────────────────────────────────────────────
function HistoryRow({ pick }: { pick: Pick }) {
  const isWin  = pick.status === 'win'
  const isLoss = pick.status === 'loss'
  const profit = pick.stake ? pick.stake * (pick.odds - 1) : 0

  return (
    <div className="hist-row">
      {/* Status bar */}
      <div className={`hist-dot${isWin ? ' win' : isLoss ? ' loss' : ''}`}/>

      {/* Match */}
      <div className="hist-match">
        <div className="hist-teams">
          {pick.leagueFlag} {pick.homeTeam} — {pick.awayTeam}
        </div>
        <div className="hist-market">{pick.side} · ×{pick.odds.toFixed(2)}</div>
      </div>

      {/* Score */}
      {pick.score && (
        <div className="hist-score">{pick.score}</div>
      )}

      {/* P&L */}
      <div className={`hist-pnl${isWin ? ' win' : isLoss ? ' loss' : ''}`}>
        {isWin  ? `+${profit.toFixed(0)}$` :
         isLoss ? (pick.stake != null ? `-${pick.stake}$` : '—') : '—'}
      </div>
    </div>
  )
}

// ── Sheet ─────────────────────────────────────────────────────────────────────
interface Props {
  model: Model
  onClose: () => void
}

export function HistorySheet({ model, onClose }: Props) {
  const [days, setDays]     = useState<HistoryDay[]>([])
  const [loading, setLoading] = useState(true)
  const [visible, setVisible] = useState(false)

  useEffect(() => { requestAnimationFrame(() => setVisible(true)) }, [])

  useEffect(() => {
    setLoading(true)
    getHistory(model).then(d => { setDays(d); setLoading(false) })
  }, [model])

  const handleClose = () => { setVisible(false); setTimeout(onClose, 300) }

  // swipe-down to close
  const startY = useRef<number | null>(null)
  const onTouchStart = (e: React.TouchEvent) => { startY.current = e.touches[0].clientY }
  const onTouchEnd   = (e: React.TouchEvent) => {
    if (startY.current !== null && e.changedTouches[0].clientY - startY.current > 60) handleClose()
    startY.current = null
  }

  // totals across all loaded days
  const totalWins   = days.reduce((s, d) => s + d.picks.filter(p => p.status === 'win').length, 0)
  const totalLosses = days.reduce((s, d) => s + d.picks.filter(p => p.status === 'loss').length, 0)
  const totalPnl    = days.reduce((s, d) => s + pnlOf(d), 0)

  return (
    <div
      className={`history-sheet-backdrop${visible ? ' visible' : ''}`}
      onClick={e => { if (e.target === e.currentTarget) handleClose() }}
    >
      <div
        className={`history-sheet${visible ? ' visible' : ''}`}
        onTouchStart={onTouchStart}
        onTouchEnd={onTouchEnd}
      >
        <div className="history-handle"/>

        {/* Header */}
        <div className="history-header">
          <button className="history-back" onClick={handleClose} aria-label="Закрити">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M19 12H5M5 12l7-7M5 12l7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <div className="history-title">Історія</div>
          <div className="history-model-badge">{model}</div>
        </div>

        {/* Summary strip */}
        {!loading && days.length > 0 && (
          <div className="history-summary">
            <div className="hs-cell">
              <div className="hs-val">{totalWins}</div>
              <div className="hs-lbl">Виграші</div>
            </div>
            <div className="hs-divider"/>
            <div className="hs-cell">
              <div className="hs-val">{totalLosses}</div>
              <div className="hs-lbl">Програші</div>
            </div>
            <div className="hs-divider"/>
            <div className="hs-cell">
              <div className={`hs-val${totalPnl >= 0 ? ' pnl-pos' : ' pnl-neg'}`}>
                {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(0)}$
              </div>
              <div className="hs-lbl">P&L</div>
            </div>
            <div className="hs-divider"/>
            <div className="hs-cell">
              <div className="hs-val">
                {totalWins + totalLosses > 0
                  ? `${Math.round(totalWins / (totalWins + totalLosses) * 100)}%`
                  : '—'}
              </div>
              <div className="hs-lbl">Win Rate</div>
            </div>
          </div>
        )}

        {/* Table header */}
        {!loading && days.length > 0 && (
          <div className="hist-thead">
            <div/>
            <div>Матч / Ставка</div>
            <div>Рах.</div>
            <div>P&L</div>
          </div>
        )}

        {/* Content */}
        <div className="history-scroll">
          {loading ? (
            <div className="history-skeleton">
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="history-skeleton-group">
                  <div className="history-skeleton-label"/>
                  {Array.from({ length: i === 1 ? 3 : i === 3 ? 1 : 2 }).map((_, j) => (
                    <div key={j} className="history-skeleton-row"/>
                  ))}
                </div>
              ))}
            </div>
          ) : days.length === 0 ? (
            <div className="history-empty">
              <span>📋</span>
              <span>Ще немає завершених матчів</span>
            </div>
          ) : (
            days.map(day => {
              const pnl    = pnlOf(day)
              const wins   = day.picks.filter(p => p.status === 'win').length
              const losses = day.picks.filter(p => p.status === 'loss').length
              return (
                <div key={day.date} className="history-day-group">
                  <div className="history-day-label">
                    <span>{formatDate(day.date)}</span>
                    <div className="history-day-meta">
                      <span className="history-day-record">
                        <span className="rec-w">{wins}W</span>
                        {' · '}
                        <span className="rec-l">{losses}L</span>
                      </span>
                      <span className={`history-day-pnl${pnl >= 0 ? ' pos' : ' neg'}`}>
                        {pnl >= 0 ? '+' : ''}{pnl.toFixed(0)}$
                      </span>
                    </div>
                  </div>
                  <div className="hist-table">
                    {day.picks.map(p => <HistoryRow key={`${day.date}-${p.id}`} pick={p}/>)}
                  </div>
                </div>
              )
            })
          )}
          <div style={{ height: 40 }}/>
        </div>
      </div>
    </div>
  )
}
