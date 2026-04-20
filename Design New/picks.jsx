// Picks screen
const { useState: useStateP } = React;

function PicksScreen({ density }) {
  const [activeDay, setActiveDay] = useStateP(2);
  const [showTop, setShowTop] = useStateP(false);
  const scrollRef = React.useRef(null);

  // 7 days centered on today (idx 2 = today)
  const today = new Date(2026, 3, 19); // Apr 19 2026
  const days = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() + (i - 2));
    const dows = ['НД','ПН','ВТ','СР','ЧТ','ПТ','СБ'];
    let dow = dows[d.getDay()];
    let narrow = null;
    if (i === 1) { narrow = 'ВЧОРА'; }
    else if (i === 2) { narrow = 'СЬОГОДНІ'; }
    else if (i === 3) { narrow = 'ЗАВТРА'; }
    return { num: d.getDate(), dow, narrow, idx: i };
  });

  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => setShowTop(el.scrollTop > 200);
    el.addEventListener('scroll', onScroll);
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  const scrollTop = () => scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });

  const picks = window.CAPPER_DATA.TODAY_PICKS;
  const compact = density === 'compact';

  return (
    <div className="page-enter">
      <div className="scroll-area" ref={scrollRef}>
        {/* Bankroll */}
        <div className="bankroll glass-strong" style={{ marginBottom: 14 }}>
          <div className="bankroll-row">
            <div>
              <div className="bankroll-label" style={{ marginBottom: 6 }}>Банкрол</div>
              <div className="bankroll-amount">
                <span className="currency">$</span>1,243<span style={{ fontSize: 22, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>.50</span>
              </div>
            </div>
            <div className="bankroll-meta">
              <div className="bankroll-roi">
                <svg width="10" height="10" viewBox="0 0 10 10"><path d="M5 1l4 5H6v3H4V6H1l4-5z" fill="currentColor"/></svg>
                ROI +24.3%
              </div>
              <div className="bankroll-label">30 ДНІВ</div>
            </div>
          </div>
          <div className="bankroll-spark">
            <Sparkline data={window.CAPPER_DATA.SPARK_DATA} height={56}/>
          </div>
        </div>

        {/* Day picker */}
        <div className="eyebrow" style={{ marginTop: 4 }}>День</div>
        <div className="day-picker" style={{ marginBottom: 14 }}>
          {days.map(d => (
            <DayChip key={d.idx} day={d}
              active={activeDay === d.idx}
              hasPicks={d.idx === 2}
              onClick={() => setActiveDay(d.idx)}/>
          ))}
        </div>

        {/* Match list */}
        <div className="eyebrow" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Матчі сьогодні</span>
          <span style={{ color: 'var(--indigo-2)' }}>{picks.length} PICKS</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {picks.map(m => <MatchCard key={m.id} m={m} compact={compact}/>)}
        </div>
      </div>

      <div className={'scroll-top' + (showTop ? ' visible' : '')} onClick={scrollTop}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M12 19V5M5 12l7-7 7 7" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
    </div>
  );
}

window.PicksScreen = PicksScreen;
