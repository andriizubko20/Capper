import { useState, useEffect, memo } from 'react'
import { COMPARE_BY_PERIOD } from '@/lib/mockData'
import type { Period, ModelData } from '@/lib/mockData'
import type { Model } from '@/lib/types'
import { getCompare } from '@/lib/api'

function winRateColor(wr: number) {
  return wr >= 65 ? 'var(--green)' : wr >= 55 ? 'var(--indigo-2)' : 'var(--red)'
}
function roiColor(roi: number) {
  return roi > 25 ? 'var(--green)' : roi >= 0 ? 'var(--indigo-2)' : 'var(--red)'
}

const MiniCurve = memo(function MiniCurve({ data, color }: { data: number[]; color: string }) {
  const w = 320, h = 60
  const min = Math.min(0, ...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const maxIndex = Math.max(data.length - 1, 1)
  const pts = data.map((v, i) => [
    (i / maxIndex) * w,
    h - ((v - min) / range) * (h - 12) - 6,
  ])
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ')
  const area = d + ` L${w},${h} L0,${h} Z`
  const gradId = `gradMini-${color.slice(1)}`
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: h, display: 'block' }} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={color} stopOpacity="0.32"/>
          <stop offset="100%" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gradId})`}/>
      <path d={d} fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
        style={{ filter: `drop-shadow(0 0 4px ${color}99)` }}/>
    </svg>
  )
})

const CombinedCurves = memo(function CombinedCurves({ models }: { models: ModelData[] }) {
  const w = 360, h = 160
  const allVals = models.flatMap(m => m.curve)
  const min = Math.min(0, ...allVals)
  const max = Math.max(...allVals)
  const range = max - min || 1
  const zeroY = h - ((0 - min) / range) * (h - 24) - 12
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: h, display: 'block' }} preserveAspectRatio="none">
      <defs>
        {models.map(m => (
          <linearGradient key={m.tag} id={`grad-${m.tag}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor={m.color} stopOpacity="0.22"/>
            <stop offset="100%" stopColor={m.color} stopOpacity="0"/>
          </linearGradient>
        ))}
      </defs>
      <line x1="0" x2={w} y1={zeroY} y2={zeroY} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 4"/>
      {models.map(m => {
        const len = m.curve.length
        const pts = m.curve.map((v, i) => [
          (i / Math.max(len - 1, 1)) * w,
          h - ((v - min) / range) * (h - 24) - 12,
        ])
        const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ')
        const area = d + ` L${w},${h} L0,${h} Z`
        const last = pts[pts.length - 1]
        return (
          <g key={m.tag}>
            <path d={area} fill={`url(#grad-${m.tag})`}/>
            <path d={d} fill="none" stroke={m.color} strokeWidth="1.8"
              strokeLinecap="round" strokeLinejoin="round"
              style={{ filter: `drop-shadow(0 0 4px ${m.color})` }}/>
            <circle cx={last[0]} cy={last[1]} r="3" fill={m.color}
              style={{ filter: `drop-shadow(0 0 6px ${m.color})` }}/>
          </g>
        )
      })}
    </svg>
  )
})

interface Props { model: Model }

export function CompareScreen({ model }: Props) {
  const [period, setPeriod] = useState<Period>('30D')
  const [models, setModels] = useState<ModelData[]>(COMPARE_BY_PERIOD[period])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    getCompare(period).then(d => { setModels(d); setLoading(false) })
  }, [period])

  return (
    <div className="scroll-area" style={{ opacity: loading ? 0.55 : 1, transition: 'opacity 0.2s' }}>
      <div className="eyebrow" style={{ marginBottom: 8 }}>Порівняння моделей · {period}</div>

      {/* Combined curves */}
      <div className="curve-card glass-strong" style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <div style={{ fontSize: 14, fontWeight: 500 }}>Profit Curves</div>
          <div className="curve-period">
            {(['7D','30D','90D','ALL'] as Period[]).map(p => (
              <div key={p} className={`period-chip${period === p ? ' active' : ''}`} onClick={() => setPeriod(p)}>{p}</div>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
          {models.map(m => (
            <div key={m.tag} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: m.color, boxShadow: `0 0 6px ${m.color}` }}/>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-dim)' }}>{m.name}</span>
            </div>
          ))}
        </div>
        <CombinedCurves models={models}/>
      </div>

      {/* Per-model cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {models.map(m => (
          <div key={m.tag} className="model-card glass" style={{
            borderColor: `${m.color}55`,
            ...(m.name === model && {
              boxShadow: `0 0 0 1.5px ${m.color}99, 0 8px 32px ${m.color}22`,
            }),
          }}>
            <div className="model-card-head">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div className="model-dot" style={{ background: m.color, boxShadow: `0 0 10px ${m.color}` }}/>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text)' }}>{m.name}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-mute)', textTransform: 'uppercase', letterSpacing: '0.1em', marginTop: 2 }}>
                    {m.avgOdds.toFixed(2)} avg odds
                  </div>
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontFamily: 'var(--font-display)', fontSize: 22, color: roiColor(m.roi), letterSpacing: '-0.3px' }}>
                  {m.roi >= 0 ? '+' : ''}{m.roi}%
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9.5, color: 'var(--text-mute)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>ROI</div>
              </div>
            </div>
            <div style={{ margin: '8px 0' }}>
              <MiniCurve data={m.curve} color={m.color}/>
            </div>
            <div className="model-stats">
              <div className="ms-cell">
                <div className="ms-val" style={{ color: winRateColor(m.win) }}>{m.win}%</div>
                <div className="ms-lbl">Win Rate</div>
              </div>
              <div className="ms-cell">
                <div className="ms-val" style={{ color: m.profit > 0 ? 'var(--green)' : 'var(--red)' }}>
                  {m.profit >= 0 ? '+' : '-'}${Math.abs(m.profit)}
                </div>
                <div className="ms-lbl">Profit</div>
              </div>
              <div className="ms-cell">
                <div className="ms-val">{m.bets}</div>
                <div className="ms-lbl">Bets</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Comparison table */}
      <div className="curve-card glass" style={{ marginTop: 14 }}>
        <div className="eyebrow" style={{ marginBottom: 10 }}>Деталі</div>

        {/* Column headers */}
        <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr 1fr 1fr', padding: '0 0 10px', borderBottom: '0.5px solid rgba(255,255,255,0.08)', gap: 6, alignItems: 'center' }}>
          <div/>
          {models.map(m => (
            <div key={m.tag} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: m.color, boxShadow: `0 0 6px ${m.color}` }}/>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>{m.name}</span>
            </div>
          ))}
        </div>

        {[
          { label: 'ROI',      vals: models.map(m => `${m.roi >= 0 ? '+' : ''}${m.roi}%`),                  getColor: (i: number) => roiColor(models[i].roi)                              },
          { label: 'Win Rate', vals: models.map(m => `${m.win}%`),                                            getColor: (i: number) => winRateColor(models[i].win)                          },
          { label: 'Profit',   vals: models.map(m => `${m.profit >= 0 ? '+' : '-'}$${Math.abs(m.profit)}`),  getColor: (i: number) => models[i].profit > 0 ? 'var(--green)' : 'var(--red)' },
          { label: 'Bets',     vals: models.map(m => `${m.bets}`),                                            getColor: () => 'var(--text)'                                                  },
          { label: 'Avg Odds', vals: models.map(m => m.avgOdds.toFixed(2)),                                   getColor: () => 'var(--text)'                                                  },
        ].map(row => (
          <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '70px 1fr 1fr 1fr', padding: '9px 0', borderBottom: '0.5px solid rgba(255,255,255,0.05)', alignItems: 'center', gap: 6 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-mute)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{row.label}</div>
            {row.vals.map((v, i) => (
              <div key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: 12, textAlign: 'center', fontWeight: 500, color: row.getColor(i) }}>{v}</div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
