import type { Pick } from '@/lib/types'
import { flagFor } from '@/lib/mockData'

const logoUrl = (id: number) => `https://media.api-sports.io/football/teams/${id}.png`

function TeamLogo({ id, name }: { id: number; name: string }) {
  return (
    <div className="team-logo-wrap">
      <img
        src={logoUrl(id)}
        alt={name}
        onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
      />
    </div>
  )
}

export function PickCard({ pick }: { pick: Pick }) {
  const isLive     = pick.status === 'live'
  const isSettled  = pick.status === 'win' || pick.status === 'loss'
  const isFinished = isSettled || pick.status === 'finished'
  const isWin      = pick.status === 'win'
  const showScore  = (isLive || isFinished) && pick.score
  const profit = pick.stake != null ? (pick.stake * (pick.odds - 1)).toFixed(0) : null

  return (
    <div className={`match-card glass${isSettled ? ` card-${pick.status}` : ''}`}>
      {/* Header */}
      <div className="match-head">
        <div className="match-meta">
          <span className="league">{(() => {
            const f = flagFor(pick.league, pick.leagueCountry)
            // FE map returned a real flag → use it; otherwise trust server-provided flag.
            return f !== '🏟' ? f : (pick.leagueFlag || '🏟')
          })()} {pick.league}</span>
          <span className="dot"/>
          <span className="match-time">{pick.time}</span>
        </div>
        {isLive && (
          <div className="live-pill">
            <div className="live-dot"/>
            LIVE
          </div>
        )}
        {pick.status === 'finished' && (
          <div className="result-pill finished">FT</div>
        )}
        {isSettled && (
          <div className={`result-pill ${pick.status}`}>
            {isWin ? (
              <>
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                  <path d="M1.5 4L3.5 6L6.5 2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                WIN
              </>
            ) : (
              <>
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                  <path d="M2 2L6 6M6 2L2 6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                </svg>
                LOSS
              </>
            )}
          </div>
        )}
        {pick.status === 'pending' && pick.timing && (
          <div className={`timing-pill ${pick.timing}`}>
            {pick.timing === 'early' ? (
              <>
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                  <circle cx="4" cy="4" r="3.25" stroke="currentColor" strokeWidth="1"/>
                  <path d="M4 2.5V4L5 5" stroke="currentColor" strokeWidth="1" strokeLinecap="round"/>
                </svg>
                РАННЯ
              </>
            ) : (
              <>
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                  <circle cx="4" cy="4" r="3.25" stroke="currentColor" strokeWidth="1"/>
                  <circle cx="4" cy="4" r="1.25" fill="currentColor"/>
                </svg>
                ФІНАЛЬНА
              </>
            )}
          </div>
        )}
      </div>

      {/* Teams */}
      <div className="match-teams">
        <div className="team home">
          <TeamLogo id={pick.homeTeamId} name={pick.homeTeam}/>
          <div className="team-name">{pick.homeTeam}</div>
        </div>
        {showScore ? (
          <div className={`score${isFinished ? ' finished' : ''}`}>{pick.score}</div>
        ) : (
          <div className="vs">vs</div>
        )}
        <div className="team away">
          <TeamLogo id={pick.awayTeamId} name={pick.awayTeam}/>
          <div className="team-name">{pick.awayTeam}</div>
        </div>
      </div>

      {/* Pick tag */}
      <div className="pick-tag-wrap">
        <div className={`pick-tag${isSettled ? ` ${pick.status}` : ''}`}>
          {pick.side}
          <span className="odds">×{pick.odds.toFixed(2)}</span>
        </div>
      </div>

      {/* Money row */}
      <div className="money-row">
        <div className="money-side">
          <div className="money-label">Ставка</div>
          <div className="money-value">{pick.stake != null ? `$${pick.stake}` : '—'}</div>
        </div>
        <div className="money-side right">
          {isSettled ? (
            <>
              <div className="money-label">{isWin ? 'Виграш' : 'Програш'}</div>
              <div className={`money-value ${pick.status}`}>
                {isWin
                  ? (profit != null ? `+$${profit}` : '—')
                  : (pick.stake != null ? `-$${pick.stake}` : '—')}
              </div>
            </>
          ) : pick.status === 'finished' ? (
            <>
              <div className="money-label">Очікуємо</div>
              <div className="money-value" style={{ color: 'var(--text-mute)' }}>—</div>
            </>
          ) : (
            <>
              <div className="money-label">Виграш</div>
              <div className="money-value win">{profit != null ? `+$${profit}` : '—'}</div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
