import type { DailyPnl } from '@/lib/mockData'

interface Props {
  date: string
  todayIso: string
  data: DailyPnl
}

function pnlColor(pnl: number) {
  return pnl > 0 ? 'var(--green)' : pnl < 0 ? 'var(--red)' : 'var(--text-dim)'
}

function dateLabel(date: string, todayIso: string) {
  const diff = Math.round(
    (new Date(date).getTime() - new Date(todayIso).getTime()) / 86400000
  )
  if (diff === 0) return 'СЬОГОДНІ'
  if (diff === -1) return 'ВЧОРА'
  return new Date(date).toLocaleDateString('uk-UA', { day: 'numeric', month: 'short' }).toUpperCase()
}

export function DailyPnlCard({ date, todayIso, data }: Props) {
  const settled = data.wins + data.losses
  const total   = settled + data.pending
  const pnlStr  = data.pnl === 0
    ? '—'
    : `${data.pnl > 0 ? '+' : '-'}$${Math.abs(data.pnl).toFixed(1)}`

  return (
    <div className="bankroll glass-strong" style={{ marginBottom: 14 }}>
      <div className="bankroll-row">
        {/* Left — P&L */}
        <div>
          <div className="bankroll-label" style={{ marginBottom: 6 }}>
            Профіт · {dateLabel(date, todayIso)}
          </div>
          <div className="bankroll-amount" style={{ color: pnlColor(data.pnl) }}>
            {pnlStr}
          </div>
        </div>

        {/* Right — W / L / pending */}
        <div className="bankroll-meta" style={{ alignItems: 'flex-end', gap: 6 }}>
          <div style={{ display: 'flex', gap: 5 }}>
            {data.wins > 0 && (
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700,
                color: 'var(--green)', background: 'rgba(34,197,94,0.14)',
                borderRadius: 999, padding: '3px 9px',
              }}>{data.wins}W</span>
            )}
            {data.losses > 0 && (
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700,
                color: 'var(--red)', background: 'rgba(239,68,68,0.14)',
                borderRadius: 999, padding: '3px 9px',
              }}>{data.losses}L</span>
            )}
            {data.pending > 0 && (
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600,
                color: 'var(--text-dim)', background: 'rgba(255,255,255,0.08)',
                borderRadius: 999, padding: '3px 9px',
              }}>{data.pending}P</span>
            )}
          </div>
          <div className="bankroll-label">${data.invested} INVESTED</div>
        </div>
      </div>

      {/* Bottom bar — matches bankroll-spark height exactly */}
      <div className="bankroll-spark" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 8 }}>
        {/* W/L/P progress bar */}
        <div style={{ height: 5, borderRadius: 999, background: 'rgba(255,255,255,0.07)', overflow: 'hidden', position: 'relative' }}>
          {total > 0 && (
            <>
              <div style={{
                position: 'absolute', left: 0, top: 0, bottom: 0,
                width: `${(data.wins / total) * 100}%`,
                background: 'var(--green)', transition: 'width 0.4s ease',
              }}/>
              <div style={{
                position: 'absolute',
                left: `${(data.wins / total) * 100}%`,
                top: 0, bottom: 0,
                width: `${(data.losses / total) * 100}%`,
                background: 'var(--red)', transition: 'all 0.4s ease',
              }}/>
            </>
          )}
        </div>

        {/* Stats row */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-mute)', letterSpacing: '0.07em' }}>
            {settled} ЗАВЕРШЕНО · {data.pending} В ОЧІКУВАННІ
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-mute)', fontWeight: 600 }}>
            {settled > 0 ? Math.round((data.wins / settled) * 100) : 0}% WR
          </span>
        </div>
      </div>
    </div>
  )
}
