import { useState, useEffect } from 'react'
import { STATS_BY_MODEL_PERIOD } from '@/lib/mockData'
import type { Period, CurvePoint, StatsData } from '@/lib/mockData'
import type { Model } from '@/lib/types'
import { getStats } from '@/lib/api'

function winRateColor(wr: number) {
  return wr >= 65 ? 'var(--green)' : wr >= 55 ? 'var(--indigo-2)' : 'var(--red)'
}
function roiColor(roi: number) {
  return roi > 25 ? 'var(--green)' : roi >= 0 ? 'var(--indigo-2)' : 'var(--red)'
}

// ─── Curve tooltip ────────────────────────────────────────────────────────────

interface TooltipState {
  idx: number
  x: number   // px from left of SVG container
  y: number   // px from top
}

function CurveTooltip({ point, x, y, containerW }: {
  point: CurvePoint
  x: number
  y: number
  containerW: number
}) {
  const w = 130
  // keep tooltip inside container horizontally
  const left = Math.min(Math.max(x - w / 2, 0), containerW - w)
  return (
    <div style={{
      position: 'absolute',
      left,
      top: Math.max(y - 72, 0),
      width: w,
      background: 'rgba(10,10,18,0.92)',
      border: '0.5px solid rgba(255,255,255,0.12)',
      borderRadius: 10,
      padding: '7px 10px',
      pointerEvents: 'none',
      zIndex: 10,
      backdropFilter: 'blur(8px)',
    }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-mute)', letterSpacing: '0.08em', marginBottom: 4 }}>
        {point.label}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: 15, color: point.profit >= 1000 ? 'var(--green)' : point.profit < 1000 ? 'var(--red)' : 'var(--text)' }}>
            ${point.profit.toFixed(0)}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--text-mute)' }}>BALANCE</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: 15, color: 'var(--text)' }}>
            {point.bets}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--text-mute)' }}>BETS</div>
        </div>
      </div>
    </div>
  )
}

// ─── Profit curve ─────────────────────────────────────────────────────────────

function ProfitCurve({
  data,
  points,
  height = 160,
}: {
  data: number[]
  points: CurvePoint[]
  height?: number
}) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const [containerW, setContainerW] = useState(360)

  const w = 360, h = height
  const min = Math.min(0, ...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const pts = data.map((v, i) => [
    (i / (data.length - 1)) * w,
    h - ((v - min) / range) * (h - 24) - 12,
  ])
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ')
  const area = d + ` L${w},${h} L0,${h} Z`
  const zeroY = h - ((0 - min) / range) * (h - 24) - 12
  const last = pts[pts.length - 1]

  const handleInteract = (clientX: number, clientY: number, rect: DOMRect) => {
    const relX = clientX - rect.left
    const relY = clientY - rect.top
    const svgX = (relX / rect.width) * w
    // find nearest point
    let nearestIdx = 0
    let nearestDist = Infinity
    pts.forEach(([px], i) => {
      const dist = Math.abs(px - svgX)
      if (dist < nearestDist) { nearestDist = dist; nearestIdx = i }
    })
    setContainerW(rect.width)
    setTooltip({ idx: nearestIdx, x: relX, y: relY })
  }

  return (
    <div
      style={{ position: 'relative' }}
      onMouseLeave={() => setTooltip(null)}
      onTouchEnd={() => setTooltip(null)}
    >
      <svg
        viewBox={`0 0 ${w} ${h}`}
        style={{ width: '100%', height, display: 'block', cursor: 'crosshair' }}
        preserveAspectRatio="none"
        onMouseMove={e => handleInteract(e.clientX, e.clientY, e.currentTarget.getBoundingClientRect())}
        onTouchMove={e => {
          e.preventDefault()
          handleInteract(e.touches[0].clientX, e.touches[0].clientY, e.currentTarget.getBoundingClientRect())
        }}
      >
        <defs>
          <linearGradient id="curveGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#7C86E0" stopOpacity="0.40"/>
            <stop offset="100%" stopColor="#7C86E0" stopOpacity="0"/>
          </linearGradient>
          <filter id="curveGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <line x1="0" x2={w} y1={zeroY} y2={zeroY} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 4"/>
        <path d={area} fill="url(#curveGrad)"/>
        <path d={d} fill="none" stroke="#7C86E0" strokeWidth="1.8" filter="url(#curveGlow)" strokeLinecap="round" strokeLinejoin="round"/>

        {/* Active point indicator */}
        {tooltip && pts[tooltip.idx] && (
          <circle
            cx={pts[tooltip.idx][0]}
            cy={pts[tooltip.idx][1]}
            r="4"
            fill="#7C86E0"
            filter="url(#curveGlow)"
          />
        )}

        {/* End dot (hidden when tooltip is near it) */}
        {(!tooltip || tooltip.idx !== pts.length - 1) && (
          <circle cx={last[0]} cy={last[1]} r="3.5" fill="#7C86E0" filter="url(#curveGlow)"/>
        )}
      </svg>

      {tooltip && points[tooltip.idx] && (
        <CurveTooltip
          point={points[tooltip.idx]}
          x={tooltip.x}
          y={tooltip.y}
          containerW={containerW}
        />
      )}
    </div>
  )
}

// ─── Screen ───────────────────────────────────────────────────────────────────

interface Props { model: Model }

export function StatsScreen({ model }: Props) {
  const [period, setPeriod] = useState<Period>('30D')
  const [data, setData] = useState<StatsData>(STATS_BY_MODEL_PERIOD[model][period])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    getStats(model, period).then(d => { setData(d); setLoading(false) })
  }, [model, period])

  const finished = data.streak.filter(s => s !== 'P').slice(-15)
  const wins     = finished.filter(s => s === 'W').length
  const losses   = finished.filter(s => s === 'L').length
  const winRate  = Math.round((wins / (wins + losses)) * 100)

  return (
    <div className="scroll-area" style={{ opacity: loading ? 0.55 : 1, transition: 'opacity 0.2s' }}>
      {/* Profit curve */}
      <div className="curve-card glass-strong" style={{ marginBottom: 12 }}>
        <div className="curve-head">
          <div>
            <div className="eyebrow" style={{ marginBottom: 6 }}>Profit Curve</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: 30, color: roiColor(data.roi), letterSpacing: '-0.5px' }}>
                {data.roi >= 0 ? '+' : ''}{data.roi}%
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)' }}>ROI</div>
            </div>
          </div>
          <div className="curve-period">
            {(['7D','30D','90D','ALL'] as Period[]).map(p => (
              <div key={p} className={`period-chip${period === p ? ' active' : ''}`} onClick={() => setPeriod(p)}>{p}</div>
            ))}
          </div>
        </div>
        <ProfitCurve data={data.curveData} points={data.curvePoints}/>
      </div>

      {/* Metrics grid */}
      <div className="stat-grid" style={{ marginBottom: 12 }}>
        <div className="stat-card glass">
          <div className="stat-value" style={{ color: roiColor(data.roi) }}>
            {data.roi >= 0 ? '+' : ''}{data.roi}%
          </div>
          <div className="stat-label">ROI</div>
        </div>
        <div className="stat-card glass">
          <div className="stat-value" style={{ color: winRateColor(data.winRate) }}>{data.winRate}%</div>
          <div className="stat-label">Win Rate</div>
        </div>
        <div className="stat-card glass">
          <div className="stat-value">{data.bets}</div>
          <div className="stat-label">Bets</div>
        </div>
        <div className="stat-card glass">
          <div className="stat-value">{data.avgOdds.toFixed(2)}</div>
          <div className="stat-label">Avg Odds</div>
        </div>
      </div>

      {/* Streak */}
      <div className="streak-card glass-strong">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div className="eyebrow" style={{ marginBottom: 4 }}>Streak</div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: 22, color: 'var(--text)' }}>
              Last {finished.length}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)' }}>
              <span style={{ color: winRateColor(winRate), fontWeight: 600 }}>{wins}W</span> / {wins + losses} bets
            </div>
          </div>
        </div>
        <div className="streak-dots">
          {finished.map((s, i) => <div key={i} className={`streak-dot ${s}`}/>)}
        </div>
        <div className="streak-legend">
          <div className="legend-bar">
            <div className="legend-bar-fill" style={{ width: `${winRate}%` }}/>
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-mute)', letterSpacing: '0.08em' }}>WIN RATE</span>
          <span style={{ color: winRateColor(winRate), fontWeight: 600 }}>{winRate}%</span>
        </div>
      </div>

      {/* By league */}
      <div className="curve-card glass" style={{ marginTop: 12 }}>
        <div className="eyebrow" style={{ marginBottom: 10 }}>By League</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: 6, borderBottom: '0.5px solid rgba(255,255,255,0.07)' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.1em', color: 'var(--text-mute)' }}>LEAGUE</span>
          <div style={{ display: 'flex', gap: 12 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.1em', color: 'var(--text-mute)', minWidth: 40, textAlign: 'right' }}>BETS</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.1em', color: 'var(--text-mute)', minWidth: 52, textAlign: 'right' }}>PROFIT</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.1em', color: 'var(--text-mute)', minWidth: 44, textAlign: 'right' }}>ROI</span>
          </div>
        </div>
        {data.byLeague.map(l => (
          <div key={l.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 0', borderBottom: '0.5px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontSize: 13, color: 'var(--text)' }}>{l.flag} {l.name}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-mute)', minWidth: 40, textAlign: 'right' }}>{l.bets}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600, color: l.profit >= 0 ? 'var(--green)' : 'var(--red)', minWidth: 52, textAlign: 'right' }}>
                {l.profit >= 0 ? '+' : '-'}${Math.abs(l.profit).toFixed(0)}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: roiColor(l.roi), minWidth: 44, textAlign: 'right', opacity: 0.7 }}>
                {l.roi >= 0 ? '+' : ''}{l.roi.toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
