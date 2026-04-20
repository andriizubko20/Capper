/**
 * Mock data layer — all types and stubs mirror the shape of future API responses.
 * When connecting the backend, replace each exported constant with an API call
 * that returns the same shape.
 *
 * API contract (future):
 *   GET /stats?model=Monster&period=7D|30D|90D|ALL  → StatsData
 *   GET /compare?period=7D|30D|90D|ALL              → ModelData[]
 *   GET /picks?date=YYYY-MM-DD&model=Monster        → Pick[]
 *   GET /daily-pnl?date=YYYY-MM-DD&model=Monster    → DailyPnl
 *   GET /bankroll                                    → BankrollData
 */

import type { Model } from '@/lib/types'

export type Period = '7D' | '30D' | '90D' | 'ALL'
export type StreakResult = 'W' | 'L' | 'P'

// ─── Stats screen ────────────────────────────────────────────────────────────

export interface LeagueRow {
  name: string
  flag: string
  bets: number
  roi: number
  profit: number
}

export interface CurvePoint {
  label: string    // human-readable date label, e.g. "Квіт 19"
  bets: number     // cumulative settled bets up to this point
  profit: number   // cumulative profit $ up to this point
}

export interface StatsData {
  roi: number
  winRate: number
  bets: number
  avgOdds: number
  curveData: number[]      // cumulative ROI % — drives the SVG line
  curvePoints: CurvePoint[] // parallel array — drives tooltip (same length as curveData)
  streak: StreakResult[]
  byLeague: LeagueRow[]
}

// ─── Curve helpers ────────────────────────────────────────────────────────────

function makeCurvePoints(
  labels: string[],
  totalBets: number,
  totalProfit: number,
  roiData: number[],
): CurvePoint[] {
  const n = labels.length
  return labels.map((label, i) => ({
    label,
    bets:   Math.round((i / (n - 1)) * totalBets),
    profit: parseFloat(((i / (n - 1)) * totalProfit).toFixed(1)),
  }))
}

const LABELS_7D  = ['Квіт 13','Квіт 14','Квіт 15','Квіт 16','Квіт 17','Квіт 18','Квіт 19']
const LABELS_30D = ['Бер 21','Бер 23','Бер 25','Бер 27','Бер 29','Бер 31','Квіт 2','Квіт 4','Квіт 6','Квіт 8','Квіт 10','Квіт 12','Квіт 14','Квіт 16','Квіт 19']
const LABELS_90D = ['Січ 19','Січ 26','Лют 2','Лют 9','Лют 16','Лют 23','Бер 2','Бер 9','Бер 16','Бер 23','Бер 30','Квіт 6','Квіт 19']
const LABELS_ALL = ['Жов','Жов 15','Лис','Лис 15','Гру','Гру 15','Січ','Січ 15','Лют','Лют 15','Бер','Бер 15','Квіт','Квіт 15']

// ─── Per-model, per-period stats ──────────────────────────────────────────────
// API: GET /stats?model=&period= → StatsData

export const STATS_BY_MODEL_PERIOD: Record<Model, Record<Period, StatsData>> = {
  'Monster': {
    '7D': {
      roi: 31.5, winRate: 78, bets: 9, avgOdds: 2.08,
      curveData:   [0, 3.5, 2.1, 6.8, 4.2, 9.1, 31.5],
      curvePoints: makeCurvePoints(LABELS_7D, 9, 124, [0,3.5,2.1,6.8,4.2,9.1,31.5]),
      streak: ['W','L','W','W','W','L','W','W','W'],
      byLeague: [
        { name: 'Premier League',  flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 3, roi: 42.1, profit: 58.4 },
        { name: 'Champions League',flag: '🏆',        bets: 3, roi: 28.7, profit: 41.2 },
        { name: 'La Liga',         flag: '🇪🇸',        bets: 2, roi: 19.4, profit: 22.8 },
        { name: 'Serie A',         flag: '🇮🇹',        bets: 1, roi: -8.5, profit: -9.1 },
      ],
    },
    '30D': {
      roi: 24.3, winRate: 61, bets: 47, avgOdds: 2.14,
      curveData: [0,1.5,0.8,2.8,3.4,4.7,5.9,6.2,7.8,9.1,10.4,11.8,12.5,13.9,24.3],
      curvePoints: makeCurvePoints(LABELS_30D, 47, 287, [0,1.5,0.8,2.8,3.4,4.7,5.9,6.2,7.8,9.1,10.4,11.8,12.5,13.9,24.3]),
      streak: ['W','W','L','W','W','W','P','W','L','W','W','W','L','W','W'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 14, roi: 28.4, profit: 114.2 },
        { name: 'La Liga',        flag: '🇪🇸',        bets: 11, roi: 19.7, profit:  78.5 },
        { name: 'Serie A',        flag: '🇮🇹',        bets:  9, roi: 22.1, profit:  63.8 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets:  8, roi: -4.2, profit: -14.7 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets:  5, roi: 11.5, profit:  27.3 },
      ],
    },
    '90D': {
      roi: 19.4, winRate: 59, bets: 134, avgOdds: 2.21,
      curveData: [0,0.8,-0.5,1.2,2.1,3.5,4.9,5.3,7.9,8.4,9.6,11.4,12.8,13.7,16.1,19.4],
      curvePoints: makeCurvePoints(LABELS_90D, 134, 842, [0,0.8,-0.5,1.2,2.1,3.5,4.9,5.3,7.9,8.4,9.6,11.4,12.8,16.1,19.4]),
      streak: ['L','W','W','L','W','P','W','W','L','W','W','L','W','L','W'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 38, roi: 24.1, profit: 312.4 },
        { name: 'La Liga',        flag: '🇪🇸',        bets: 29, roi: 18.3, profit: 201.5 },
        { name: 'Serie A',        flag: '🇮🇹',        bets: 27, roi: 15.7, profit: 178.3 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets: 24, roi: 21.2, profit: 241.8 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets: 16, roi: -3.8, profit: -44.2 },
      ],
    },
    'ALL': {
      roi: 22.4, winRate: 60, bets: 287, avgOdds: 2.18,
      curveData: [0,-0.8,1.4,3.2,5.4,7.1,8.9,10.2,11.8,13.2,14.7,15.4,17.8,19.2,20.4,21.7,22.4],
      curvePoints: makeCurvePoints(LABELS_ALL, 287, 2241, [0,-0.8,1.4,3.2,5.4,7.1,8.9,10.2,11.8,13.2,14.7,15.4,17.8,19.2,20.4,21.7,22.4]),
      streak: ['W','L','W','W','P','L','W','W','W','L','W','W','L','W','W'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 82, roi: 26.3, profit: 718.4 },
        { name: 'La Liga',        flag: '🇪🇸',        bets: 71, roi: 20.1, profit: 521.8 },
        { name: 'Serie A',        flag: '🇮🇹',        bets: 58, roi: 18.4, profit: 412.7 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets: 47, roi: 24.9, profit: 387.2 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets: 29, roi:  9.7, profit: 128.3 },
      ],
    },
  },

  'WS Gap': {
    '7D': {
      roi: 12.1, winRate: 56, bets: 9, avgOdds: 2.18,
      curveData:   [0, 1.2, 0.4, 2.8, 2.1, 4.5, 12.1],
      curvePoints: makeCurvePoints(LABELS_7D, 9, 48, [0,1.2,0.4,2.8,2.1,4.5,12.1]),
      streak: ['W','L','L','W','W','L','W','W','L'],
      byLeague: [
        { name: 'Premier League',  flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 4, roi: 18.2, profit: 24.1 },
        { name: 'La Liga',         flag: '🇪🇸',        bets: 3, roi:  9.4, profit: 12.8 },
        { name: 'Bundesliga',      flag: '🇩🇪',        bets: 2, roi: -5.2, profit: -7.4 },
      ],
    },
    '30D': {
      roi: 18.4, winRate: 56, bets: 42, avgOdds: 2.31,
      curveData: [0,1.1,0.4,2.2,1.8,3.5,4.2,5.1,6.4,7.8,8.5,9.2,10.4,12.1,18.4],
      curvePoints: makeCurvePoints(LABELS_30D, 42, 218, [0,1.1,0.4,2.2,1.8,3.5,4.2,5.1,6.4,7.8,8.5,9.2,10.4,12.1,18.4]),
      streak: ['L','W','W','L','W','W','P','L','W','W','L','W','W','L','W'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 12, roi: 22.1, profit: 84.2 },
        { name: 'La Liga',        flag: '🇪🇸',        bets: 10, roi: 15.3, profit: 58.4 },
        { name: 'Serie A',        flag: '🇮🇹',        bets:  8, roi: 18.7, profit: 52.1 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets:  7, roi: 12.4, profit: 37.8 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets:  5, roi: -6.8, profit: -24.1 },
      ],
    },
    '90D': {
      roi: 22.7, winRate: 58, bets: 118, avgOdds: 2.28,
      curveData: [0,0.9,-0.3,1.8,2.5,4.1,5.7,6.2,8.4,9.8,10.5,12.3,13.7,15.1,18.2,19.4,21.9,22.7],
      curvePoints: makeCurvePoints(LABELS_90D, 118, 714, [0,0.9,-0.3,1.8,2.5,4.1,5.7,6.2,8.4,9.8,10.5,12.3,13.7,15.1,18.2,19.4,21.9,22.7]),
      streak: ['W','L','W','W','L','W','P','W','L','W','W','W','L','W','W'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 34, roi: 28.4, profit: 284.2 },
        { name: 'La Liga',        flag: '🇪🇸',        bets: 28, roi: 21.7, profit: 198.4 },
        { name: 'Serie A',        flag: '🇮🇹',        bets: 24, roi: 19.2, profit: 162.8 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets: 20, roi: 24.8, profit: 181.4 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets: 12, roi: 14.2, profit: 76.1 },
      ],
    },
    'ALL': {
      roi: 25.3, winRate: 59, bets: 254, avgOdds: 2.26,
      curveData: [0,-1.1,1.2,2.8,4.2,6.1,7.9,9.3,11.2,13.1,14.7,16.2,17.8,19.4,21.1,22.4,23.9,25.3],
      curvePoints: makeCurvePoints(LABELS_ALL, 254, 1842, [0,-1.1,1.2,2.8,4.2,6.1,7.9,9.3,11.2,13.1,14.7,16.2,17.8,19.4,21.1,22.4,23.9,25.3]),
      streak: ['W','W','L','W','W','P','W','L','W','W','W','L','W','W','W'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 74, roi: 29.1, profit: 648.2 },
        { name: 'La Liga',        flag: '🇪🇸',        bets: 62, roi: 23.4, profit: 471.8 },
        { name: 'Serie A',        flag: '🇮🇹',        bets: 54, roi: 21.8, profit: 384.1 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets: 42, roi: 27.2, profit: 348.9 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets: 22, roi: 12.8, profit: 104.2 },
      ],
    },
  },

  'Aqua': {
    '7D': {
      roi: 8.4, winRate: 44, bets: 9, avgOdds: 2.51,
      curveData:   [0, 0.8, -1.2, 1.4, 0.2, 2.8, 8.4],
      curvePoints: makeCurvePoints(LABELS_7D, 9, 32, [0,0.8,-1.2,1.4,0.2,2.8,8.4]),
      streak: ['L','L','W','L','W','W','L','W','W'],
      byLeague: [
        { name: 'Premier League',  flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 3, roi: 12.4, profit: 14.2 },
        { name: 'Champions League',flag: '🏆',        bets: 3, roi:  4.8, profit:  8.1 },
        { name: 'Serie A',         flag: '🇮🇹',        bets: 2, roi: -4.2, profit: -6.1 },
        { name: 'Ligue 1',         flag: '🇫🇷',        bets: 1, roi: 14.8, profit:  9.8 },
      ],
    },
    '30D': {
      roi: 11.7, winRate: 52, bets: 38, avgOdds: 2.48,
      curveData: [0,0.8,-0.4,1.2,0.7,2.1,1.5,3.2,2.8,4.5,3.9,5.7,5.2,7.4,11.7],
      curvePoints: makeCurvePoints(LABELS_30D, 38, 142, [0,0.8,-0.4,1.2,0.7,2.1,1.5,3.2,2.8,4.5,3.9,5.7,5.2,7.4,11.7]),
      streak: ['L','W','L','W','W','L','P','W','L','W','W','L','W','W','L'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 11, roi: 14.2, profit: 48.4 },
        { name: 'La Liga',        flag: '🇪🇸',        bets:  9, roi:  8.7, profit: 28.1 },
        { name: 'Serie A',        flag: '🇮🇹',        bets:  8, roi: 12.4, profit: 36.8 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets:  6, roi: -7.8, profit: -21.4 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets:  4, roi:  9.1, profit: 18.4 },
      ],
    },
    '90D': {
      roi: 14.2, winRate: 54, bets: 106, avgOdds: 2.44,
      curveData: [0,0.5,-0.8,0.9,1.7,1.2,2.8,3.4,4.5,5.1,6.2,7.4,8.1,9.2,10.4,11.7,13.2,14.2],
      curvePoints: makeCurvePoints(LABELS_90D, 106, 521, [0,0.5,-0.8,0.9,1.7,1.2,2.8,3.4,4.5,5.1,6.2,7.4,8.1,9.2,10.4,11.7,13.2,14.2]),
      streak: ['W','L','L','W','W','P','L','W','W','L','W','W','L','W','L'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 30, roi: 17.8, profit: 184.2 },
        { name: 'La Liga',        flag: '🇪🇸',        bets: 24, roi: 12.4, profit: 108.4 },
        { name: 'Serie A',        flag: '🇮🇹',        bets: 22, roi: 11.2, profit:  94.8 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets: 18, roi: 16.4, profit: 112.1 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets: 12, roi:  4.8, profit:  28.4 },
      ],
    },
    'ALL': {
      roi: 16.8, winRate: 55, bets: 231, avgOdds: 2.41,
      curveData: [0,-0.5,0.9,1.8,3.1,2.7,4.5,5.8,7.1,8.4,9.7,10.8,11.9,13.1,14.2,15.4,16.8],
      curvePoints: makeCurvePoints(LABELS_ALL, 231, 1124, [0,-0.5,0.9,1.8,3.1,2.7,4.5,5.8,7.1,8.4,9.7,10.8,11.9,13.1,14.2,15.4,16.8]),
      streak: ['L','W','W','L','W','W','P','W','L','W','L','W','W','W','L'],
      byLeague: [
        { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', bets: 68, roi: 19.2, profit: 428.4 },
        { name: 'La Liga',        flag: '🇪🇸',        bets: 54, roi: 14.8, profit: 284.1 },
        { name: 'Serie A',        flag: '🇮🇹',        bets: 48, roi: 13.4, profit: 238.7 },
        { name: 'Bundesliga',     flag: '🇩🇪',        bets: 38, roi: 18.7, profit: 248.2 },
        { name: 'Ligue 1',        flag: '🇫🇷',        bets: 23, roi:  8.4, profit:  84.1 },
      ],
    },
  },
}

// ─── Compare screen ───────────────────────────────────────────────────────────

export interface ModelData {
  name: string
  tag: string
  color: string
  roi: number
  win: number
  bets: number
  avgOdds: number
  profit: number
  curve: number[]
}

const MODEL_META = {
  gap:     { name: 'WS Gap' as Model,  tag: 'gap',     color: '#F43F8E' },
  monster: { name: 'Monster' as Model, tag: 'monster',  color: '#FACC15' },
  aqua:    { name: 'Aqua' as Model,    tag: 'aqua',     color: '#22D3EE' },
}

export const MODEL_COLOR: Record<Model, string> = {
  'WS Gap':  MODEL_META.gap.color,
  'Monster': MODEL_META.monster.color,
  'Aqua':    MODEL_META.aqua.color,
}

export const COMPARE_BY_PERIOD: Record<Period, ModelData[]> = {
  '7D': [
    { ...MODEL_META.gap,     roi: 12.1, win: 56, bets: 9,  avgOdds: 2.18, profit:  48, curve: [0,1.2,0.4,2.8,2.1,4.5,6.2,9.4,12.1] },
    { ...MODEL_META.monster, roi: 31.5, win: 78, bets: 9,  avgOdds: 2.08, profit: 124, curve: [0,3.5,2.1,6.8,4.2,9.1,15.4,22.8,31.5] },
    { ...MODEL_META.aqua,    roi:  8.4, win: 44, bets: 9,  avgOdds: 2.51, profit:  32, curve: [0,0.8,-1.2,1.4,0.2,2.8,1.9,4.7,8.4] },
  ],
  '30D': [
    { ...MODEL_META.gap,     roi: 18.4, win: 56, bets: 42,  avgOdds: 2.31, profit:  218, curve: [0,1.1,0.4,2.2,1.8,3.5,4.2,3.8,5.1,6.4,5.9,7.8,8.5,9.2,10.4,12.1,11.5,13.2,14.5,15.8,16.4,17.1,18.4] },
    { ...MODEL_META.monster, roi: 24.3, win: 61, bets: 47,  avgOdds: 2.14, profit:  287, curve: [0,1.5,0.8,2.8,3.4,4.7,5.9,6.2,7.8,9.1,10.4,11.8,12.5,13.9,15.2,16.7,17.8,19.4,20.5,21.7,22.8,23.5,24.3] },
    { ...MODEL_META.aqua,    roi: 11.7, win: 52, bets: 38,  avgOdds: 2.48, profit:  142, curve: [0,0.8,-0.4,1.2,0.7,2.1,1.5,3.2,2.8,4.5,3.9,5.7,5.2,6.8,7.4,8.1,8.8,9.5,10.2,10.7,11.1,11.4,11.7] },
  ],
  '90D': [
    { ...MODEL_META.gap,     roi: 22.7, win: 58, bets: 118, avgOdds: 2.28, profit:  714, curve: [0,0.9,-0.3,1.8,2.5,4.1,3.8,6.2,5.7,8.4,7.9,10.5,9.8,12.3,13.7,12.9,15.1,14.4,16.8,18.2,17.5,19.4,20.8,19.7,21.9,22.7] },
    { ...MODEL_META.monster, roi: 19.4, win: 59, bets: 134, avgOdds: 2.21, profit:  842, curve: [0,0.8,-0.5,1.2,0.4,2.1,3.5,2.8,4.2,5.7,4.9,6.8,5.3,7.9,9.2,8.4,10.1,9.6,11.4,12.8,11.9,13.5,14.8,13.7,16.1,19.4] },
    { ...MODEL_META.aqua,    roi: 14.2, win: 54, bets: 106, avgOdds: 2.44, profit:  521, curve: [0,0.5,-0.8,0.9,0.3,1.7,1.2,2.8,2.1,3.9,3.4,5.1,4.7,6.2,7.4,6.8,8.1,7.5,9.2,10.4,9.8,11.3,12.1,11.7,13.2,14.2] },
  ],
  'ALL': [
    { ...MODEL_META.gap,     roi: 25.3, win: 59, bets: 254, avgOdds: 2.26, profit: 1842, curve: [0,-1.1,1.2,0.4,2.8,4.2,3.5,6.1,5.4,7.9,9.3,8.6,11.2,10.5,13.1,14.7,13.9,16.2,15.5,17.8,19.4,18.7,21.1,22.4,21.8,23.9,25.3] },
    { ...MODEL_META.monster, roi: 22.4, win: 60, bets: 287, avgOdds: 2.18, profit: 2241, curve: [0,-0.8,1.4,0.8,3.2,2.1,5.4,4.8,7.1,6.3,8.9,7.5,10.2,9.4,11.8,13.2,12.5,14.7,13.9,16.1,15.4,17.8,19.2,18.5,20.4,21.7,22.4] },
    { ...MODEL_META.aqua,    roi: 16.8, win: 55, bets: 231, avgOdds: 2.41, profit: 1124, curve: [0,-0.5,0.9,0.2,1.8,1.4,3.1,2.7,4.5,3.9,5.8,5.2,7.1,6.5,8.4,9.7,9.1,10.8,10.2,11.9,13.1,12.5,14.2,13.8,15.4,16.8] },
  ],
}

// ─── Daily P&L ────────────────────────────────────────────────────────────────
// API: GET /daily-pnl?date=YYYY-MM-DD&model= → DailyPnl

export interface DailyPnl {
  pnl: number
  wins: number
  losses: number
  pending: number
  invested: number
}

export const DAILY_PNL: Record<string, DailyPnl> = {
  '2026-04-17': { pnl: -31.5, wins: 1, losses: 3, pending: 0, invested: 145 },
  '2026-04-18': { pnl:  82.4, wins: 4, losses: 1, pending: 0, invested: 180 },
  '2026-04-19': { pnl:  17.5, wins: 1, losses: 1, pending: 2, invested: 185 },
  '2026-04-20': { pnl:   0,   wins: 0, losses: 0, pending: 1, invested:  55 },
  '2026-04-21': { pnl:   0,   wins: 0, losses: 0, pending: 1, invested:  40 },
}

// ─── Bankroll ─────────────────────────────────────────────────────────────────
// API: GET /bankroll → BankrollData

export interface BankrollData {
  amount: number
  roi: number
}

export const BANKROLL: BankrollData = {
  amount: 1243.50,
  roi: 24.3,
}
