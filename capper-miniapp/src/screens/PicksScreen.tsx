import { useState, useRef, useEffect } from 'react'
import { BankrollCard } from '@/components/BankrollCard'
import { DailyPnlCard } from '@/components/DailyPnlCard'
import { PickCard } from '@/components/PickCard'
import { HistorySheet } from '@/components/HistorySheet'
import { STATS_BY_MODEL_PERIOD } from '@/lib/mockData'
import type { StatsData, DailyPnl } from '@/lib/mockData'
import type { Pick, Model } from '@/lib/types'
import { getPicks, getStats } from '@/lib/api'

const TODAY = new Date()

function isoDate(d: Date) { return d.toISOString().slice(0, 10) }
function addDays(d: Date, n: number) { const r = new Date(d); r.setDate(r.getDate() + n); return r }

const DAYS_BACK = 5
const DAYS_FORWARD = 5
const TOTAL = DAYS_BACK + 1 + DAYS_FORWARD
const TODAY_ISO = isoDate(TODAY)
const DOW_UK = ['НД', 'ПН', 'ВТ', 'СР', 'ЧТ', 'ПТ', 'СБ']

function buildDays() {
  return Array.from({ length: TOTAL }, (_, i) => {
    const d = addDays(TODAY, i - DAYS_BACK)
    const diff = i - DAYS_BACK
    const narrow = diff === -1 ? 'ВЧОРА' : diff === 0 ? 'СЬОГОДНІ' : diff === 1 ? 'ЗАВТРА' : null
    return { iso: isoDate(d), num: d.getDate(), dow: DOW_UK[d.getDay()], narrow }
  })
}

const DAYS = buildDays()

// ─── Swipeable hero card ──────────────────────────────────────────────────────

interface SwipeableProps {
  card0: React.ReactNode
  card1: React.ReactNode
}

function SwipeableCards({ card0, card1 }: SwipeableProps) {
  const [idx, setIdx] = useState(0)
  const [drag, setDrag] = useState(0)
  const startX = useRef<number | null>(null)
  const dragging = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── auto-swipe every 10s; resets when user swipes manually ──
  const resetTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = setInterval(() => setIdx(i => i === 0 ? 1 : 0), 10_000)
  }
  useEffect(() => {
    resetTimer()
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [])

  const startDrag = (x: number) => { startX.current = x; dragging.current = true }
  const moveDrag  = (x: number) => {
    if (!dragging.current || startX.current === null) return
    const delta = x - startX.current
    if ((idx === 0 && delta < 0) || (idx === 1 && delta > 0)) setDrag(delta)
  }
  const endDrag = () => {
    if (Math.abs(drag) > 50) { setIdx(drag < 0 ? 1 : 0); resetTimer() }
    setDrag(0)
    startX.current = null
    dragging.current = false
  }

  // ── touch ──
  const onTouchStart = (e: React.TouchEvent) => startDrag(e.touches[0].clientX)
  const onTouchMove  = (e: React.TouchEvent) => moveDrag(e.touches[0].clientX)
  const onTouchEnd   = () => endDrag()

  // ── mouse (for browser preview) ──
  const onMouseDown  = (e: React.MouseEvent) => startDrag(e.clientX)
  const onMouseMove  = (e: React.MouseEvent) => moveDrag(e.clientX)
  const onMouseUp    = () => endDrag()
  const onMouseLeave = () => { if (dragging.current) endDrag() }

  const w = containerRef.current?.offsetWidth || 390
  const translate = -idx * 100 + (drag / w) * 100

  return (
    <div style={{ marginBottom: 6 }}>
      <div ref={containerRef} style={{ overflow: 'hidden' }}>
        <div
          style={{
            display: 'flex',
            transform: `translateX(${translate}%)`,
            transition: drag !== 0 ? 'none' : 'transform 0.32s cubic-bezier(0.4,0,0.2,1)',
            willChange: 'transform',
            userSelect: 'none',
            cursor: drag !== 0 ? 'grabbing' : 'grab',
          }}
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
          onTouchEnd={onTouchEnd}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseLeave}
        >
          <div style={{ minWidth: '100%' }}>{card0}</div>
          <div style={{ minWidth: '100%' }}>{card1}</div>
        </div>
      </div>

      {/* Dots */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 5, marginTop: 8, marginBottom: 6 }}>
        {[0, 1].map(i => (
          <div
            key={i}
            onClick={() => { setIdx(i); resetTimer() }}
            style={{
              height: 4, borderRadius: 999, cursor: 'pointer',
              width: i === idx ? 18 : 4,
              background: i === idx ? 'var(--indigo-2)' : 'rgba(255,255,255,0.18)',
              transition: 'all 0.25s ease',
            }}
          />
        ))}
      </div>
    </div>
  )
}

// ─── Screen ───────────────────────────────────────────────────────────────────

interface Props { model: Model }

export function PicksScreen({ model }: Props) {
  const [activeDay, setActiveDay] = useState(TODAY_ISO)
  const [showTop, setShowTop] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const dayPickerRef = useRef<HTMLDivElement>(null)
  const [picks, setPicks] = useState<Pick[]>([])
  const [picksLoading, setPicksLoading] = useState(false)
  const [modelStats, setModelStats] = useState<StatsData>(STATS_BY_MODEL_PERIOD[model]['30D'])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = () => setShowTop(el.scrollTop > 200)
    el.addEventListener('scroll', onScroll)
    return () => el.removeEventListener('scroll', onScroll)
  }, [])

  // Scroll day picker to active chip
  useEffect(() => {
    const picker = dayPickerRef.current
    if (!picker) return
    const active = picker.querySelector('.day-chip.active') as HTMLElement | null
    if (!active) return
    const chipCenter = active.offsetLeft + active.offsetWidth / 2
    picker.scrollTo({ left: chipCenter - picker.offsetWidth / 2, behavior: 'smooth' })
  }, [activeDay])

  // Fetch model stats for bankroll card when model changes
  useEffect(() => {
    let cancelled = false
    getStats(model, '30D').then(s => { if (!cancelled) setModelStats(s) })
    return () => { cancelled = true }
  }, [model])

  // Fetch picks when day or model changes
  // cancelled flag prevents stale responses from overwriting newer data
  useEffect(() => {
    let cancelled = false
    setPicksLoading(true)
    getPicks(activeDay, model).then(p => {
      if (!cancelled) { setPicks(p); setPicksLoading(false) }
    }).catch(() => {
      if (!cancelled) setPicksLoading(false)
    })
    return () => { cancelled = true }
  }, [activeDay, model])

  const scrollTop = () => scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' })

  const refresh = () => {
    setPicksLoading(true)
    getPicks(activeDay, model).then(p => { setPicks(p); setPicksLoading(false) })
    getStats(model, '30D').then(setModelStats)
  }

  // Рахуємо daily P&L з завантажених picks
  const daily: DailyPnl | null = picks.length === 0 ? null : {
    pnl:     picks.reduce((s, p) => s + (p.pnl ?? 0), 0),
    wins:    picks.filter(p => p.status === 'win').length,
    losses:  picks.filter(p => p.status === 'loss').length,
    pending: picks.filter(p => p.status === 'pending' || p.status === 'live' || p.status === 'finished').length,
    invested: Math.round(picks.reduce((s, p) => s + (p.stake ?? 0), 0)),
  }

  return (
    <>
      <div className="scroll-area" ref={scrollRef}>

        {/* Hero — swipe between bankroll and daily P&L */}
        <SwipeableCards
          card0={
            <BankrollCard
              amount={modelStats.curveData.length ? modelStats.curveData[modelStats.curveData.length - 1] : 1000}
              roi={modelStats.roi}
              sparkline={modelStats.curveData}
            />
          }
          card1={<DailyPnlCard date={activeDay} todayIso={TODAY_ISO} data={daily}/>}
        />

        <div className="eyebrow" style={{ marginTop: 4, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span>День</span>
          <button
            className="history-btn"
            onClick={() => setShowHistory(true)}
            aria-label="Відкрити історію матчів"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9.25" stroke="currentColor" strokeWidth="1.6"/>
              <path d="M12 7v5l3.5 2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3.5 8.5A9 9 0 0 1 12 3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
              <path d="M3.5 8.5 2 6.5M3.5 8.5 5.5 7" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
            </svg>
            ІСТОРІЯ
          </button>
        </div>
        <div className="day-picker" style={{ marginBottom: 14 }} ref={dayPickerRef}>
          {DAYS.map(d => {
            const isActive = d.iso === activeDay
            const isNamed = !!d.narrow
            return (
              <div
                key={d.iso}
                className={`day-chip${isActive ? ' active' : ''}${isNamed ? ' named' : ''}`}
                onClick={() => setActiveDay(d.iso)}
              >
                <div className="dp-dow">{d.narrow || d.dow}</div>
                <div className="dp-num">{d.num}</div>
                {d.iso === TODAY_ISO && <div className="dp-dot"/>}
              </div>
            )
          })}
        </div>

        <div className="eyebrow" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Матчі</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ color: 'var(--indigo-2)' }}>{picksLoading ? '…' : `${picks.length} PICKS`}</span>
            <button
              onClick={refresh}
              disabled={picksLoading}
              style={{
                background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                color: picksLoading ? 'var(--text-mute)' : 'var(--text-dim)',
                display: 'flex', alignItems: 'center',
                transition: 'color 0.2s',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                style={{ animation: picksLoading ? 'spin 0.8s linear infinite' : 'none' }}>
                <path d="M4 12a8 8 0 0 1 14.93-4M20 12a8 8 0 0 1-14.93 4"
                  stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
                <path d="M19 4v4h-4M5 20v-4h4" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
        </div>

        {/* Initial load — show skeleton */}
        {picksLoading && picks.length === 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[1, 2, 3].map(i => (
              <div key={i} style={{ height: 110, borderRadius: 16, background: 'rgba(255,255,255,0.05)' }}/>
            ))}
          </div>
        )}

        {/* Has picks — keep them visible while loading new ones */}
        {picks.length > 0 && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: 10,
            opacity: picksLoading ? 0.5 : 1,
            transition: 'opacity 0.25s',
            pointerEvents: picksLoading ? 'none' : 'auto',
          }}>
            {picks.map(p => <PickCard key={p.id} pick={p}/>)}
          </div>
        )}

        {/* Empty state */}
        {!picksLoading && picks.length === 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, padding: '48px 0', color: 'var(--text-mute)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
            <span style={{ fontSize: 32 }}>📭</span>
            Немає пікс на цей день
          </div>
        )}
      </div>

      <button
        className={`scroll-top-btn${showTop ? ' visible' : ''}`}
        onClick={scrollTop}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M12 19V5M5 12l7-7 7 7" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {showHistory && (
        <HistorySheet model={model} onClose={() => setShowHistory(false)}/>
      )}
    </>
  )
}
