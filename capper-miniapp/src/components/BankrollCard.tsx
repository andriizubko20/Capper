function Sparkline({ data }: { data: number[] }) {
  if (data.length === 0) return <svg className="spark-svg"/>
  const w = 320, h = 56
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const maxIndex = Math.max(data.length - 1, 1)
  const pts = data.map((v, i) => [
    (i / maxIndex) * w,
    h - ((v - min) / range) * (h - 8) - 4,
  ])
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ')
  const area = d + ` L${w},${h} L0,${h} Z`
  const last = pts[pts.length - 1]
  return (
    <svg className="spark-svg" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#7C86E0" stopOpacity="0.35"/>
          <stop offset="100%" stopColor="#7C86E0" stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path d={area} className="spark-area"/>
      <path d={d} className="spark-line"/>
      <circle cx={last[0]} cy={last[1]} r="3" className="spark-end"/>
    </svg>
  )
}

const FALLBACK_SPARK = [1000, 1000, 1000]

interface Props {
  amount: number
  roi: number
  sparkline?: number[]
  period?: string
}

export function BankrollCard({ amount, roi, sparkline, period = '30 ДНІВ' }: Props) {
  const dollars = Math.floor(Math.abs(amount))
  const cents = (Math.abs(amount) % 1).toFixed(2).slice(1)
  const positive = roi >= 0
  const roiColor = roi > 0 ? 'var(--green)' : roi < 0 ? 'var(--red)' : 'var(--text-dim)'

  return (
    <div className="bankroll glass-strong" style={{ marginBottom: 14 }}>
      <div className="bankroll-row">
        <div>
          <div className="bankroll-label" style={{ marginBottom: 6 }}>Банкрол</div>
          <div className="bankroll-amount">
            <span className="currency">$</span>
            {dollars.toLocaleString()}
            <span style={{ fontSize: 22, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
              {cents}
            </span>
          </div>
        </div>
        <div className="bankroll-meta">
          <div className="bankroll-roi" style={{ color: roiColor, borderColor: `${roiColor}44`, background: `${roiColor}14` }}>
            <svg width="10" height="10" viewBox="0 0 10 10"
              style={{ transform: positive ? 'none' : 'rotate(180deg)', transition: 'transform 0.2s' }}>
              <path d="M5 1l4 5H6v3H4V6H1l4-5z" fill="currentColor"/>
            </svg>
            ROI {positive ? '+' : ''}{roi}%
          </div>
          <div className="bankroll-label">{period}</div>
        </div>
      </div>
      <div className="bankroll-spark">
        <Sparkline data={sparkline && sparkline.length > 1 ? sparkline : FALLBACK_SPARK}/>
      </div>
    </div>
  )
}
