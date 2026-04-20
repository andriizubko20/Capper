// Compare screen — model comparison
function CompareScreen() {
  const D = window.CAPPER_DATA;

  const models = [
    {
      name: 'WC Gap', tag: 'gap', color: '#7C86E0',
      roi: 18.4, win: 56, bets: 42, avgOdds: 2.31, profit: 218,
      curve: [0, 1.1, 0.4, 2.2, 1.8, 3.5, 4.2, 3.8, 5.1, 6.4, 5.9, 7.8, 8.5, 9.2, 10.4, 12.1, 11.5, 13.2, 14.5, 15.8, 16.4, 17.1, 18.4],
    },
    {
      name: 'Monster', tag: 'monster', color: '#22C55E',
      roi: 24.3, win: 61, bets: 47, avgOdds: 2.14, profit: 287,
      curve: [0, 1.5, 0.8, 2.8, 3.4, 4.7, 5.9, 6.2, 7.8, 9.1, 10.4, 11.8, 12.5, 13.9, 15.2, 16.7, 17.8, 19.4, 20.5, 21.7, 22.8, 23.5, 24.3],
    },
    {
      name: 'Aqua', tag: 'aqua', color: '#22D3EE',
      roi: 11.7, win: 52, bets: 38, avgOdds: 2.48, profit: 142,
      curve: [0, 0.8, -0.4, 1.2, 0.7, 2.1, 1.5, 3.2, 2.8, 4.5, 3.9, 5.7, 5.2, 6.8, 7.4, 8.1, 8.8, 9.5, 10.2, 10.7, 11.1, 11.4, 11.7],
    },
  ];

  // mini line for each model
  const Mini = ({ data, color }) => {
    const w = 320, h = 60;
    const min = Math.min(0, ...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const pts = data.map((v, i) => [
      (i / (data.length - 1)) * w,
      h - ((v - min) / range) * (h - 12) - 6,
    ]);
    const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
    const area = d + ` L${w},${h} L0,${h} Z`;
    return (
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: h, display: 'block' }} preserveAspectRatio="none">
        <defs>
          <linearGradient id={`gradMini-${color.slice(1)}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.32"/>
            <stop offset="100%" stopColor={color} stopOpacity="0"/>
          </linearGradient>
        </defs>
        <path d={area} fill={`url(#gradMini-${color.slice(1)})`}/>
        <path d={d} fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
          style={{ filter: `drop-shadow(0 0 4px ${color}99)` }}/>
      </svg>
    );
  };

  // Best metric per column
  const best = {
    roi: Math.max(...models.map(m => m.roi)),
    win: Math.max(...models.map(m => m.win)),
    profit: Math.max(...models.map(m => m.profit)),
  };

  return (
    <div className="page-enter">
      <div className="scroll-area">
        <div className="eyebrow" style={{ marginBottom: 8 }}>Порівняння моделей · 30 ДНІВ</div>

        {/* Combined curves */}
        <div className="curve-card glass-strong" style={{ marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 12 }}>
            <div style={{ fontSize: 14, fontWeight: 500 }}>Profit Curves</div>
            <div style={{ display: 'flex', gap: 10 }}>
              {models.map(m => (
                <div key={m.tag} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 50, background: m.color, boxShadow: `0 0 6px ${m.color}` }}/>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-dim)' }}>{m.name}</span>
                </div>
              ))}
            </div>
          </div>
          <div style={{ position: 'relative', height: 160 }}>
            <CombinedCurves models={models}/>
          </div>
        </div>

        {/* Per-model cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {models.map(m => (
            <div key={m.tag} className="model-card glass" style={{
              borderColor: `${m.color}55`,
            }}>
              <div className="model-card-head">
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div className="model-dot" style={{ background: m.color, boxShadow: `0 0 10px ${m.color}` }}/>
                  <div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text)' }}>{m.name}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-mute)', textTransform: 'uppercase', letterSpacing: '0.1em', marginTop: 2 }}>{m.bets} bets · {m.avgOdds.toFixed(2)} avg</div>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{
                    fontFamily: 'var(--font-display)', fontSize: 22, color: m.color,
                    textShadow: m.roi === best.roi ? `0 0 12px ${m.color}80` : 'none',
                    letterSpacing: '-0.3px',
                  }}>+{m.roi}%</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9.5, color: 'var(--text-mute)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>ROI</div>
                </div>
              </div>
              <div style={{ margin: '8px 0' }}>
                <Mini data={m.curve} color={m.color}/>
              </div>
              <div className="model-stats">
                <div className="ms-cell">
                  <div className={'ms-val' + (m.win === best.win ? ' best' : '')}>{m.win}%</div>
                  <div className="ms-lbl">Win Rate</div>
                </div>
                <div className="ms-cell">
                  <div className={'ms-val' + (m.profit === best.profit ? ' best' : '')}>+${m.profit}</div>
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

        {/* Side-by-side comparison table */}
        <div className="curve-card glass" style={{ marginTop: 14 }}>
          <div className="eyebrow" style={{ marginBottom: 10 }}>Деталі</div>
          {[
            { label: 'ROI', vals: models.map(m => `+${m.roi}%`), bestIdx: models.findIndex(m => m.roi === best.roi) },
            { label: 'Win Rate', vals: models.map(m => `${m.win}%`), bestIdx: models.findIndex(m => m.win === best.win) },
            { label: 'Profit', vals: models.map(m => `+$${m.profit}`), bestIdx: models.findIndex(m => m.profit === best.profit) },
            { label: 'Bets', vals: models.map(m => `${m.bets}`), bestIdx: -1 },
            { label: 'Avg Odds', vals: models.map(m => m.avgOdds.toFixed(2)), bestIdx: -1 },
          ].map(row => (
            <div key={row.label} style={{
              display: 'grid', gridTemplateColumns: '70px 1fr 1fr 1fr',
              padding: '8px 0', borderBottom: '0.5px solid rgba(255,255,255,0.05)',
              alignItems: 'center', gap: 6,
            }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-mute)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{row.label}</div>
              {row.vals.map((v, i) => (
                <div key={i} style={{
                  fontFamily: 'var(--font-mono)', fontSize: 12,
                  textAlign: 'center', fontWeight: row.bestIdx === i ? 600 : 500,
                  color: row.bestIdx === i ? models[i].color : 'var(--text)',
                  textShadow: row.bestIdx === i ? `0 0 8px ${models[i].color}50` : 'none',
                }}>{v}</div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function CombinedCurves({ models }) {
  const w = 360, h = 160;
  const allVals = models.flatMap(m => m.curve);
  const min = Math.min(0, ...allVals);
  const max = Math.max(...allVals);
  const range = max - min || 1;
  const len = models[0].curve.length;
  const zeroY = h - ((0 - min) / range) * (h - 24) - 12;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: h, display: 'block' }} preserveAspectRatio="none">
      <defs>
        {models.map(m => (
          <linearGradient key={m.tag} id={`grad-${m.tag}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={m.color} stopOpacity="0.22"/>
            <stop offset="100%" stopColor={m.color} stopOpacity="0"/>
          </linearGradient>
        ))}
      </defs>
      <line x1="0" x2={w} y1={zeroY} y2={zeroY} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 4"/>
      {models.map(m => {
        const pts = m.curve.map((v, i) => [
          (i / (len - 1)) * w,
          h - ((v - min) / range) * (h - 24) - 12,
        ]);
        const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
        const area = d + ` L${w},${h} L0,${h} Z`;
        const last = pts[pts.length - 1];
        return (
          <g key={m.tag}>
            <path d={area} fill={`url(#grad-${m.tag})`}/>
            <path d={d} fill="none" stroke={m.color} strokeWidth="1.8"
              strokeLinecap="round" strokeLinejoin="round"
              style={{ filter: `drop-shadow(0 0 4px ${m.color})` }}/>
            <circle cx={last[0]} cy={last[1]} r="3" fill={m.color}
              style={{ filter: `drop-shadow(0 0 6px ${m.color})` }}/>
          </g>
        );
      })}
    </svg>
  );
}

window.CompareScreen = CompareScreen;
