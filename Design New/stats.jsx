// Statistics screen
function StatsScreen() {
  const [period, setPeriod] = React.useState('30D');
  const D = window.CAPPER_DATA;

  const wins = D.STREAK.filter(s => s === 'W').length;
  const losses = D.STREAK.filter(s => s === 'L').length;
  const pushes = D.STREAK.filter(s => s === 'P').length;
  const winRate = (wins / (wins + losses)) * 100;

  return (
    <div className="page-enter">
      <div className="scroll-area">
        {/* Profit curve */}
        <div className="curve-card glass-strong" style={{ marginBottom: 12 }}>
          <div className="curve-head">
            <div>
              <div className="eyebrow" style={{ marginBottom: 6 }}>Profit Curve</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <div style={{ fontFamily: 'var(--font-display)', fontSize: 30, color: 'var(--green)', letterSpacing: '-0.5px' }}>+24.3%</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)' }}>ROI</div>
              </div>
            </div>
            <div className="curve-period">
              {['7D','30D','90D','ALL'].map(p => (
                <div key={p}
                  className={'period-chip' + (period === p ? ' active' : '')}
                  onClick={() => setPeriod(p)}>{p}</div>
              ))}
            </div>
          </div>
          <ProfitCurve data={D.CURVE_DATA}/>
        </div>

        {/* Metrics grid */}
        <div className="stat-grid" style={{ marginBottom: 12 }}>
          <div className="stat-card glass">
            <div className="stat-value green">+24.3%</div>
            <div className="stat-label">ROI</div>
          </div>
          <div className="stat-card glass">
            <div className="stat-value indigo">61%</div>
            <div className="stat-label">Win Rate</div>
          </div>
          <div className="stat-card glass">
            <div className="stat-value">47</div>
            <div className="stat-label">Bets</div>
          </div>
          <div className="stat-card glass">
            <div className="stat-value">2.14</div>
            <div className="stat-label">Avg Odds</div>
          </div>
        </div>

        {/* Streak */}
        <div className="streak-card glass-strong">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div className="eyebrow" style={{ marginBottom: 4 }}>Streak</div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: 22, color: 'var(--text)' }}>
                Останні 15
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)' }}>
                <span style={{ color: 'var(--indigo-2)', fontWeight: 600 }}>{wins}W</span> / {wins+losses+pushes} bets
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-mute)', marginTop: 2 }}>
                {losses}L · {pushes}P
              </div>
            </div>
          </div>
          <div className="streak-dots">
            {D.STREAK.map((s, i) => (
              <div key={i} className={'streak-dot ' + s} title={s}/>
            ))}
          </div>
          <div className="streak-legend">
            <span style={{ color: 'var(--indigo-2)', fontWeight: 600 }}>{wins}W</span>
            <div className="legend-bar">
              <div className="legend-bar-fill" style={{ width: `${(wins/15)*100}%` }}/>
            </div>
            <span>{wins} / 14 bets</span>
          </div>
        </div>

        {/* By league snippet */}
        <div className="curve-card glass" style={{ marginTop: 12 }}>
          <div className="eyebrow" style={{ marginBottom: 10 }}>By League</div>
          {[
            { name: 'Premier League', bets: 14, roi: 28.4 },
            { name: 'La Liga', bets: 11, roi: 19.7 },
            { name: 'Serie A', bets: 9, roi: 22.1 },
            { name: 'Bundesliga', bets: 8, roi: -4.2 },
            { name: 'Ligue 1', bets: 5, roi: 11.5 },
          ].map(l => (
            <div key={l.name} style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '8px 0', borderBottom: '0.5px solid rgba(255,255,255,0.05)',
            }}>
              <div style={{ fontSize: 13, color: 'var(--text)' }}>{l.name}</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-mute)' }}>{l.bets} bets</span>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600,
                  color: l.roi >= 0 ? 'var(--green)' : 'var(--red)',
                  minWidth: 56, textAlign: 'right',
                }}>{l.roi >= 0 ? '+' : ''}{l.roi.toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

window.StatsScreen = StatsScreen;
