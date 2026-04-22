export type Screen = 'picks' | 'stats' | 'compare'
export type Model  = 'WS Gap' | 'Monster' | 'Aqua'
export type PickStatus = 'live' | 'win' | 'loss' | 'pending' | 'finished'
export type BetSide = 'HOME' | 'AWAY' | 'DRAW' | `OVER ${string}` | `UNDER ${string}`

export type BetTiming = 'early' | 'final'

// Telegram WebApp global injected by Telegram client
declare global {
  interface Window {
    Telegram?: { WebApp?: { initData: string } }
  }
}

export interface Pick {
  id: string
  model: Model
  league: string
  leagueFlag: string
  homeTeam: string
  awayTeam: string
  homeTeamId: number
  awayTeamId: number
  time: string
  side: BetSide
  odds: number
  ev: number
  stake: number
  status: PickStatus
  timing?: BetTiming
  pnl?: number
  score?: string
  date: 'today' | 'yesterday' | string
}

export interface ScheduleMatch {
  id: string
  league: string
  leagueFlag: string
  homeTeam: string
  awayTeam: string
  homeTeamId: number
  awayTeamId: number
  time: string
  date: string
  hasPick?: boolean
  pickSide?: string
  pickOdds?: number
}
