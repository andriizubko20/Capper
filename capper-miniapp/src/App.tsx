import { useState, useCallback } from 'react'
import { BottomNav } from '@/components/BottomNav'
import { LoadingScreen } from '@/components/LoadingScreen'
import { PicksScreen } from '@/screens/PicksScreen'
import { StatsScreen } from '@/screens/StatsScreen'
import { CompareScreen } from '@/screens/CompareScreen'
import type { Screen, Model } from '@/lib/types'

const MODELS: Model[] = ['WS Gap', 'Monster', 'Aqua', 'Pure', 'Gem']

export default function App() {
  const [screen, setScreen] = useState<Screen>('picks')
  const [model, setModel] = useState<Model>('Monster')
  const [loading, setLoading] = useState(true)

  const handleLoadingDone = useCallback(() => setLoading(false), [])

  return (
    <div className="capper-root">
      <div className="aurora"><div className="blob3"/></div>
      <div className="grain"/>

      {/* Model switcher bar */}
      <div className="tg-bar">
        <div className="model-switch">
          {MODELS.map(m => (
            <button
              key={m}
              className={`model-btn${model === m ? ' active' : ''}`}
              onClick={() => setModel(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Screen */}
      <main key={screen} className="page-enter">
        {screen === 'picks'   && <PicksScreen model={model}/>}
        {screen === 'stats'   && <StatsScreen model={model}/>}
        {screen === 'compare' && <CompareScreen model={model}/>}
      </main>

      <BottomNav active={screen} onChange={setScreen}/>

      {/* Splash screen — shown on first mount */}
      {loading && <LoadingScreen onDone={handleLoadingDone}/>}
    </div>
  )
}
