// Capper — shared UI primitives
const { useState, useEffect, useRef, useMemo } = React;
const D = window.CAPPER_DATA;

// Crest — colored circle with team initials
function Crest({ code, size = 26 }) {
  const team = D.TEAMS[code] || { name: code, color: '#444' };
  const initials = team.name.split(' ').map(w => w[0]).join('').slice(0, 3).toUpperCase();
  return (
    <div className="team-crest" style={{
      width: size, height: size,
      background: `linear-gradient(135deg, ${team.color} 0%, ${shade(team.color, -25)} 100%)`,
    }}>{initials}</div>
  );
}
function shade(hex, pct) {
  const num = parseInt(hex.slice(1), 16);
  let r = (num >> 16) + pct;
  let g = ((num >> 8) & 0xff) + pct;
  let b = (num & 0xff) + pct;
  r = Math.max(0, Math.min(255, r));
  g = Math.max(0, Math.min(255, g));
  b = Math.max(0, Math.min(255, b));
  return '#' + ((r << 16) | (g << 8) | b).toString(16).padStart(6, '0');
}

// Sparkline
function Sparkline({ data, height = 56, gradient = true }) {
  const w = 320, h = height;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => [
    (i / (data.length - 1)) * w,
    h - ((v - min) / range) * (h - 8) - 4,
  ]);
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  const area = d + ` L${w},${h} L0,${h} Z`;
  const last = pts[pts.length - 1];
  return (
    <svg className="spark-svg" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#7C86E0" stopOpacity="0.35"/>
          <stop offset="100%" stopColor="#7C86E0" stopOpacity="0"/>
        </linearGradient>
      </defs>
      {gradient && <path d={area} className="spark-area" />}
      <path d={d} className="spark-line" />
      <circle cx={last[0]} cy={last[1]} r="3" className="spark-end" />
    </svg>
  );
}

// Profit curve — bigger, with axis hints
function ProfitCurve({ data, height = 160 }) {
  const w = 360, h = height;
  const min = Math.min(0, ...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => [
    (i / (data.length - 1)) * w,
    h - ((v - min) / range) * (h - 24) - 12,
  ]);
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  const area = d + ` L${w},${h} L0,${h} Z`;
  const zeroY = h - ((0 - min) / range) * (h - 24) - 12;
  const last = pts[pts.length - 1];
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height, display: 'block' }} preserveAspectRatio="none">
      <defs>
        <linearGradient id="curveGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#7C86E0" stopOpacity="0.40"/>
          <stop offset="100%" stopColor="#7C86E0" stopOpacity="0"/>
        </linearGradient>
        <filter id="curveGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="2" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {/* zero line */}
      <line x1="0" x2={w} y1={zeroY} y2={zeroY} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 4"/>
      <path d={area} fill="url(#curveGrad)"/>
      <path d={d} fill="none" stroke="#7C86E0" strokeWidth="1.8" filter="url(#curveGlow)"
        strokeLinecap="round" strokeLinejoin="round"/>
      <circle cx={last[0]} cy={last[1]} r="3.5" fill="#7C86E0" filter="url(#curveGlow)"/>
    </svg>
  );
}

// Telegram chrome bar — with model switcher
function TelegramBar({ model, onModelChange }) {
  return (
    <div className="tg-bar">
      <div className="tg-close">Закрити</div>
      <div className="model-switch">
        {['WC Gap','Monster','Aqua'].map(m => (
          <button key={m}
            className={'model-btn' + (model === m ? ' active' : '')}
            onClick={() => onModelChange(m)}>{m}</button>
        ))}
      </div>
      <div className="tg-more">
        <svg width="18" height="4" viewBox="0 0 18 4">
          <circle cx="2" cy="2" r="1.6" fill="#8A8A93"/>
          <circle cx="9" cy="2" r="1.6" fill="#8A8A93"/>
          <circle cx="16" cy="2" r="1.6" fill="#8A8A93"/>
        </svg>
      </div>
    </div>
  );
}

// Bottom nav
function BottomNav({ active, onChange }) {
  const items = [
    { id: 'picks', label: 'Schedule', icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="5" width="18" height="16" rx="3" stroke="currentColor" strokeWidth="1.8"/>
        <path d="M3 9h18M8 3v4M16 3v4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
      </svg>
    )},
    { id: 'stats', label: 'Statistics', icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <path d="M4 19V8M10 19V4M16 19v-7M22 19H2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
      </svg>
    )},
    { id: 'compare', label: 'Compare', icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <path d="M8 3v18M16 3v18M3 8h5M3 16h5M16 8h5M16 16h5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
      </svg>
    )},
  ];
  return (
    <div className="bottom-nav">
      {items.map(it => (
        <div key={it.id}
          className={'nav-item' + (active === it.id ? ' active' : '')}
          onClick={() => onChange(it.id)}>
          {it.icon}
          <div className="nav-label">{it.label}</div>
        </div>
      ))}
    </div>
  );
}

// Day picker chip pill
function DayChip({ day, active, hasPicks, onClick }) {
  const isNamed = !!day.narrow;
  return (
    <div className={'day-chip' + (active ? ' active' : '') + (isNamed ? ' named' : '')} onClick={onClick}>
      <div className="dp-dow">{day.narrow || day.dow}</div>
      <div className="dp-num">{day.num}</div>
      {hasPicks && <div className="dp-dot"/>}
    </div>
  );
}

// Match card
function MatchCard({ m, compact = false }) {
  const win = (m.stake * (m.odds - 1)).toFixed(0);
  return (
    <div className={'match-card glass'}>
      <div className="match-head">
        <div className="match-meta">
          <span className="league">{m.league}</span>
          <span className="dot"/>
          <span className="match-time">{m.live ? m.minute : m.time}</span>
        </div>
        {m.live && (
          <div className="live-pill">
            <div className="live-dot"/>
            LIVE
          </div>
        )}
      </div>
      <div className="match-teams">
        <div className="team home">
          <Crest code={m.home} size={28}/>
          <div className="team-name">{D.TEAMS[m.home].name}</div>
        </div>
        {m.live && m.score ? (
          <div className="score">{m.score}</div>
        ) : (
          <div className="vs">vs</div>
        )}
        <div className="team away">
          <Crest code={m.away} size={28}/>
          <div className="team-name">{D.TEAMS[m.away].name}</div>
        </div>
      </div>
      <div className="pick-tag-wrap">
        <div className="pick-tag">
          {m.pick}
          <span className="odds">×{m.odds.toFixed(2)}</span>
        </div>
      </div>
      <div className="money-row">
        <div className="money-side">
          <div className="money-label">Ставка</div>
          <div className="money-value">${m.stake}</div>
        </div>
        <div className="money-side right">
          <div className="money-label">Виграш</div>
          <div className="money-value win">+${win}</div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, {
  Crest, Sparkline, ProfitCurve, TelegramBar, BottomNav, DayChip, MatchCard,
});
