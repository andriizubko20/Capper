/**
 * API client — typed fetch wrappers for all backend endpoints.
 *
 * Set VITE_API_URL in .env to point at the FastAPI server, e.g.:
 *   VITE_API_URL=http://localhost:8000
 *
 * If VITE_API_URL is not set or request fails → falls back to mock data.
 */

import type { Pick, Model } from '@/lib/types'
import {
  STATS_BY_MODEL_PERIOD,
  COMPARE_BY_PERIOD,
  BANKROLL,
  type Period,
  type StatsData,
  type ModelData,
  type BankrollData,
} from '@/lib/mockData'

const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '')

// ─── helpers ─────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

// ─── Picks ────────────────────────────────────────────────────────────────────
// GET /api/picks?date=YYYY-MM-DD&model=Monster → { picks: Pick[] }

// Конвертує UTC ISO рядок у локальний час телефону: "14:30"
function utcIsoToLocalTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
  } catch {
    return iso.slice(11, 16) // fallback: просто обрізаємо
  }
}

export async function getPicks(date: string, model: Model): Promise<Pick[]> {
  if (!BASE) return []
  try {
    const data = await apiFetch<{ picks: (Pick & { datetime_utc?: string })[] }>(
      `/api/picks?date=${date}&model=${encodeURIComponent(model)}`
    )
    const picks = data.picks ?? []
    // Якщо бекенд повернув datetime_utc — конвертуємо в локальний час
    return picks.map(p =>
      // Для live матчів time вже містить хвилину ('42\'') — не перезаписуємо
      p.datetime_utc && p.status !== 'live'
        ? { ...p, time: utcIsoToLocalTime(p.datetime_utc) }
        : p
    )
  } catch {
    return []
  }
}

// ─── Stats ────────────────────────────────────────────────────────────────────
// GET /api/stats?model=Monster&period=30d → StatsData (mapped)

interface ApiStats {
  roi: number
  winRate: number
  profit: number
  balance: number           // поточний баланс = 1000 + total_pnl
  startingBalance: number   // завжди 1000
  totalBets: number
  avgOdds: number
  streak: { result: string; id: string }[]
  byLeague: { league: string; country?: string; flag: string; bets: number; winRate: number; pnl: number; roi: number }[]
  curve: number[]           // bankroll curve: починається з ~1000
}

function mapApiStats(api: ApiStats, fallback: StatsData): StatsData {
  // Якщо немає даних — повертаємо mock
  if (!api.totalBets) return fallback

  const streak = api.streak.map(s =>
    s.result === 'win' ? 'W' : s.result === 'loss' ? 'L' : 'P'
  ) as StatsData['streak']

  const byLeague = api.byLeague.map(l => ({
    name:    l.league,
    country: l.country,
    flag:    l.flag,
    bets:    l.bets,
    roi:     l.roi,
    profit:  l.pnl,
  }))

  // API curve = bankroll (e.g. 1050, 980…). Convert to PnL$ by subtracting start.
  const start = api.startingBalance ?? 1000
  const pnlCurve = [0, ...api.curve.map(v => parseFloat((v - start).toFixed(2)))]
  const n = pnlCurve.length
  const curvePoints: StatsData['curvePoints'] = pnlCurve.map((val, i) => ({
    label:  i === 0 ? 'Start' : `Bet ${i}`,
    bets:   Math.round((i / Math.max(n - 1, 1)) * api.totalBets),
    profit: val,
  }))

  return {
    roi:        api.roi,
    winRate:    api.winRate,
    bets:       api.totalBets,
    avgOdds:    api.avgOdds,
    curveData:  pnlCurve,
    curvePoints,
    streak,
    byLeague,
  }
}

export async function getStats(model: Model, period: Period): Promise<StatsData> {
  const fallback = STATS_BY_MODEL_PERIOD[model][period]
  if (!BASE) return fallback
  try {
    const api = await apiFetch<ApiStats>(
      `/api/stats?model=${encodeURIComponent(model)}&period=${period.toLowerCase()}`
    )
    return mapApiStats(api, fallback)
  } catch {
    return fallback
  }
}

// ─── Compare ─────────────────────────────────────────────────────────────────
// GET /api/compare?period=30d → { models: ApiModel[] }

interface ApiModel {
  name: string
  color: string
  roi: number
  winRate: number
  bets: number
  avgOdds: number
  profit: number
  curve: number[]
}

const MODEL_TAG: Record<string, string> = {
  'WS Gap':  'gap',
  'Monster': 'monster',
  'Aqua':    'aqua',
  'Pure':    'pure',
  'Gem':     'gem',
  'Gem v2':  'gem_v2',
}

export async function getCompare(period: Period): Promise<ModelData[]> {
  const fallback = COMPARE_BY_PERIOD[period]
  if (!BASE) return fallback
  try {
    const data = await apiFetch<{ models: ApiModel[] }>(
      `/api/compare?period=${period.toLowerCase()}`
    )
    if (!data.models?.length) return fallback
    return data.models.map(m => ({
      name:     m.name,
      tag:      MODEL_TAG[m.name] ?? m.name.toLowerCase(),
      color:    m.color,
      roi:      m.roi,
      win:      m.winRate,
      bets:     m.bets,
      avgOdds:  m.avgOdds,
      profit:   m.profit,
      curve:    m.curve.length ? m.curve : [0],
    }))
  } catch {
    return fallback
  }
}

// ─── History ─────────────────────────────────────────────────────────────────
// GET /api/history?model=Monster → { dates: { date, picks }[] }

export interface HistoryDay { date: string; picks: Pick[] }

export async function getHistory(model: Model): Promise<HistoryDay[]> {
  if (!BASE) return []
  try {
    const data = await apiFetch<{ dates: HistoryDay[] }>(
      `/api/history?model=${encodeURIComponent(model)}`
    )
    return data.dates ?? []
  } catch {
    return []
  }
}

// ─── CLV ──────────────────────────────────────────────────────────────────────
// GET /api/clv?model=Gem&days=30 → ClvResponse

export interface ClvTrendPoint {
  date:    string  // ISO date (YYYY-MM-DD)
  avg_clv: number  // fraction, e.g. 0.014 = +1.4 %
  n:       number  // picks on that date
}

export interface ClvResponse {
  model:    string
  days:     number
  avg_clv:  number          // fraction
  n_picks:  number
  pos_rate: number          // fraction of picks with CLV > 0
  trend:    ClvTrendPoint[]
}

export async function getClv(model: Model, days = 30): Promise<ClvResponse | null> {
  if (!BASE) return null
  try {
    return await apiFetch<ClvResponse>(
      `/api/clv?model=${encodeURIComponent(model)}&days=${days}`
    )
  } catch {
    return null
  }
}

// ─── Bankroll ─────────────────────────────────────────────────────────────────
// GET /api/bankroll → { balance, roi, sparkline }

export async function getBankroll(): Promise<BankrollData> {
  if (!BASE) return BANKROLL
  try {
    // Pass Telegram WebApp initData for server-side signature verification
    const initData = window.Telegram?.WebApp?.initData
    const query = initData ? `?init_data=${encodeURIComponent(initData)}` : ''
    const data = await apiFetch<{ balance: number; roi: number; sparkline: number[] }>(
      `/api/bankroll${query}`
    )
    return { amount: data.balance, roi: data.roi }
  } catch {
    return BANKROLL
  }
}
