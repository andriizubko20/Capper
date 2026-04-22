const logo = (id: number) => `https://media.api-sports.io/football/teams/${id}.png`

interface Match {
  id: string; league: string; homeTeam: string; awayTeam: string
  homeTeamId: number; awayTeamId: number; time: string
  hasPick?: boolean; pickSide?: string; pickOdds?: number
}
interface Day { label: string; date: string; matches: Match[] }

const SCHEDULE: Day[] = [
  {
    label: 'Сьогодні', date: '18 кв',
    matches: [
      { id:'s1', league:'Premier League', homeTeam:'Arsenal', awayTeam:'Liverpool', homeTeamId:42, awayTeamId:40, time:'14:00', hasPick:true, pickSide:'AWAY', pickOdds:2.45 },
      { id:'s2', league:'Champions League', homeTeam:'Bayern', awayTeam:'PSG', homeTeamId:157, awayTeamId:85, time:'20:00', hasPick:true, pickSide:'HOME', pickOdds:1.95 },
      { id:'s3', league:'Bundesliga', homeTeam:'Dortmund', awayTeam:'Leipzig', homeTeamId:165, awayTeamId:173, time:'17:30' },
    ],
  },
  {
    label: 'Завтра', date: '19 кв',
    matches: [
      { id:'s4', league:'La Liga', homeTeam:'Atletico', awayTeam:'Sevilla', homeTeamId:530, awayTeamId:536, time:'16:00' },
      { id:'s5', league:'Serie A', homeTeam:'Inter', awayTeam:'Napoli', homeTeamId:505, awayTeamId:492, time:'19:45', hasPick:true, pickSide:'OVER 2.5', pickOdds:1.72 },
    ],
  },
  {
    label: 'Вт', date: '20 кв',
    matches: [
      { id:'s6', league:'Ligue 1', homeTeam:'Marseille', awayTeam:'Monaco', homeTeamId:81, awayTeamId:91, time:'20:00' },
      { id:'s7', league:'UCL', homeTeam:'Real Madrid', awayTeam:'Man City', homeTeamId:541, awayTeamId:50, time:'21:00' },
    ],
  },
]

function MatchRow({ m }: { m: Match }) {
  return (
    <div className="glass-raised rounded-xl flex items-center px-3.5 py-3 mb-1.5 min-h-[56px] cursor-pointer
      transition-all duration-150 ease-spring hover:border-white/[0.14] active:scale-[0.985]">

      {/* logo pair */}
      <div className="flex items-center flex-shrink-0 mr-3">
        {[m.homeTeamId, m.awayTeamId].map((id, i) => (
          <div key={i} className={`w-6 h-6 rounded-full bg-white/[0.05] border border-white/[0.08] overflow-hidden flex-shrink-0 ${i>0?'-ml-2':''}`}>
            <img src={logo(id)} alt="" className="w-full h-full object-contain p-0.5"
              onError={e=>{(e.currentTarget as HTMLImageElement).style.display='none'}}/>
          </div>
        ))}
      </div>

      {/* info */}
      <div className="flex-1 min-w-0">
        <p className="text-[9px] font-medium text-capper-dim mb-0.5 uppercase tracking-wide">{m.league}</p>
        <p className="text-[13px] font-semibold text-capper-text truncate tracking-[-0.01em]">
          {m.homeTeam} <span className="text-capper-dim font-normal">–</span> {m.awayTeam}
        </p>
      </div>

      {/* right */}
      <div className="flex flex-col items-end gap-1.5 ml-3 flex-shrink-0">
        <span className="font-mono text-[11px] text-capper-muted tabular">{m.time}</span>
        {m.hasPick && (
          <span className="inline-flex items-center gap-1 text-[9px] font-mono font-semibold px-2 py-0.5 rounded-md"
            style={{background:'rgba(94,106,210,0.14)',border:'1px solid rgba(94,106,210,0.28)',color:'#5E6AD2'}}>
            {m.pickSide} · ×{m.pickOdds}
          </span>
        )}
      </div>
    </div>
  )
}

export function ScheduleScreen() {
  return (
    <div className="flex flex-col h-full overflow-y-auto scrollbar-none px-4 pt-2 pb-[180px]">
      {SCHEDULE.map(day => (
        <div key={day.label} className="mb-1">
          <div className="flex items-center gap-2 py-2.5 flex-shrink-0">
            <span className="text-[10px] font-bold tracking-[0.12em] text-capper-dim uppercase">{day.label}</span>
            <span className="font-mono text-[10px] text-capper-dim/50">{day.date}</span>
            <span className="text-[9px] font-mono text-capper-indigo/70 bg-capper-indigo/10 border border-capper-indigo/20 px-1.5 py-0.5 rounded-md">
              {day.matches.filter(m=>m.hasPick).length > 0
                ? `${day.matches.filter(m=>m.hasPick).length} pick${day.matches.filter(m=>m.hasPick).length>1?'s':''}`
                : `${day.matches.length} matches`}
            </span>
            <div className="flex-1 h-px bg-gradient-to-r from-white/[0.07] to-transparent"/>
          </div>
          {day.matches.map(m => <MatchRow key={m.id} m={m}/>)}
        </div>
      ))}
      <div className="h-2 flex-shrink-0"/>
    </div>
  )
}
