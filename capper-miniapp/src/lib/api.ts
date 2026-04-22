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
  byLeague: { league: string; flag: string; bets: number; winRate: number; pnl: number }[]
  curve: number[]           // bankroll curve: починається з ~1000
}

function mapApiStats(api: ApiStats, fallback: StatsData): StatsData {
  // Якщо немає даних — повертаємо mock
  if (!api.totalBets) return fallback

  const streak = api.streak.map(s =>
    s.result === 'win' ? 'W' : s.result === 'loss' ? 'L' : 'P'
  ) as StatsData['streak']

  const byLeague = api.byLeague.map(l => ({
    name:   l.league,
    flag:   l.flag,
    bets:   l.bets,
    roi:    l.winRate,   // winRate як proxy для roi до окремого endpoint
    profit: l.pnl,
  }))

  // curvePoints — bankroll від $1000 за кожною ставкою
  const n = api.curve.length
  const curvePoints = api.curve.map((val, i) => ({
    label:  `Bet ${i + 1}`,
    bets:   Math.round((i / Math.max(n - 1, 1)) * api.totalBets),
    profit: val,   // тепер це bankroll (починається з ~1000)
  }))

  return {
    roi:        api.roi,
    winRate:    api.winRate,
    bets:       api.totalBets,
    avgOdds:    api.avgOdds,
    curveData:  api.curve,
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

// ─── Bankroll ─────────────────────────────────────────────────────────────────
// GET /api/bankroll → { balance, roi, sparkline }

export async function getBankroll(): Promise<BankrollData> {
  if (!BASE) return BANKROLL
  try {
    const data = await apiFetch<{ balance: number; roi: number; sparkline: number[] }>(
      '/api/bankroll'
    )
    return { amount: data.balance, roi: data.roi }
  } catch {
    return BANKROLL
  }
}
