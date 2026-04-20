import { useEffect, useRef, useState } from 'react'

interface Props { onDone: () => void }

// ── Gerstner wave components (physically-based ocean surface) ─────────────────
// Each component: amplitude, wavelength, phase offset, direction, speed
const GERSTNER = [
  { A: 22, len: 260, ph: 0,   dir:  1, spd: 0.20 },  // primary swell
  { A: 13, len: 145, ph: 1.4, dir:  1, spd: 0.27 },  // secondary
  { A:  8, len:  85, ph: 2.6, dir: -1, spd: 0.35 },  // cross-wave
  { A:  3, len:  42, ph: 0.8, dir:  1, spd: 0.46 },  // chop
  { A: 1.5,len:  20, ph: 3.8, dir: -1, spd: 0.62 },  // ripple
]

function surfaceH(x: number, t: number): number {
  let h = 0
  for (const w of GERSTNER) {
    const k = (2 * Math.PI) / w.len
    h += w.A * Math.cos(k * x * w.dir - w.spd * t + w.ph)
  }
  return h
}

interface Particle {
  x: number; y: number
  vx: number; vy: number
  r: number; life: number; maxLife: number
}

// ── Reveal text ───────────────────────────────────────────────────────────────
function RevealText() {
  return (
    <div style={{
      position: 'absolute', inset: 0, display: 'flex',
      flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      animation: 'textReveal 0.9s cubic-bezier(0.16,1,0.3,1) forwards',
      pointerEvents: 'none',
    }}>
      <div style={{
        position: 'absolute', width: 380, height: 220,
        background: 'radial-gradient(ellipse, rgba(94,106,210,0.3) 0%, transparent 70%)',
        filter: 'blur(32px)',
      }}/>
      <div style={{
        fontFamily: 'var(--font-display)', fontSize: 50,
        letterSpacing: '0.03em', color: 'var(--text)', lineHeight: 1,
        marginBottom: 12, position: 'relative',
      }}>
        Aqua Predict
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 11,
        letterSpacing: '0.22em', color: 'var(--text-mute)',
        textTransform: 'uppercase', position: 'relative',
      }}>
        AI · Football · Predictions
      </div>
    </div>
  )
}

// ── Main screen ───────────────────────────────────────────────────────────────
export function LoadingScreen({ onDone }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [showText, setShowText] = useState(false)
  const [out, setOut]           = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    const CW = canvas.width  = window.innerWidth
    const CH = canvas.height = window.innerHeight

    // Animation state — all mutable, lives in closure
    let time = 0, lastTs = -1, phaseT = 0
    let particles: Particle[] = []
    let doneFired = false

    // Phase sequence
    const PHASES = [
      { name: 'rising',  dur: 1.50 },
      { name: 'impact',  dur: 0.55 },
      { name: 'falling', dur: 1.10 },
      { name: 'reveal',  dur: 1.40 },
      { name: 'out',     dur: 0.40 },
    ] as const
    let pi = 0  // phase index

    const easeOut = (t: number) => 1 - (1 - t) ** 3
    const easeIn  = (t: number) => t ** 3

    const BOT = CH + 220
    const TOP = -CH * 0.24   // wave goes past top so it fills screen

    function baseY(): number {
      const p = Math.min(phaseT / PHASES[pi].dur, 1)
      switch (PHASES[pi].name) {
        case 'rising':  return BOT + (TOP - BOT) * easeOut(p)
        case 'impact':  return TOP
        case 'falling': return TOP + (BOT - TOP) * easeIn(p)
        default:        return BOT
      }
    }

    function spawnFoam() {
      for (let i = 0; i < 55; i++) {
        const ml = 0.45 + Math.random() * 0.7
        particles.push({
          x:  Math.random() * CW,
          y:  Math.random() * 18,
          vx: (Math.random() - 0.5) * 260,
          vy: 25 + Math.random() * 160,
          r:  2.5 + Math.random() * 7.5,
          life: ml, maxLife: ml,
        })
      }
    }

    function frame(ts: number) {
      if (lastTs < 0) lastTs = ts
      const dt = Math.min((ts - lastTs) / 1000, 0.033)
      lastTs = ts
      time  += dt
      phaseT += dt

      // Phase transition
      if (phaseT >= PHASES[pi].dur && pi < PHASES.length - 1) {
        pi++
        phaseT = 0
        if (PHASES[pi].name === 'impact')  spawnFoam()
        if (PHASES[pi].name === 'reveal')  setShowText(true)
        if (PHASES[pi].name === 'out')     setOut(true)
        if (PHASES[pi].name === 'out' && !doneFired) {
          doneFired = true
          setTimeout(onDone, 400)
        }
      }

      ctx.clearRect(0, 0, CW, CH)

      // Draw wave only during water phases
      const wname = PHASES[pi].name
      if (wname === 'rising' || wname === 'impact' || wname === 'falling') {
        const by = baseY()

        // Surface points (every 2px for performance)
        const pts: [number, number][] = []
        for (let x = 0; x <= CW; x += 2) pts.push([x, by + surfaceH(x, time)])

        // Find min surface Y for gradient start
        let minY = pts[0][1]
        for (const p of pts) if (p[1] < minY) minY = p[1]

        // ── Water body ──
        ctx.beginPath()
        ctx.moveTo(0, CH + 10)
        for (const [x, y] of pts) ctx.lineTo(x, y)
        ctx.lineTo(CW, CH + 10)
        ctx.closePath()

        const wg = ctx.createLinearGradient(0, minY, 0, CH)
        wg.addColorStop(0,    'rgba(210,214,248,0.93)')
        wg.addColorStop(0.06, 'rgba(165,173,238,0.91)')
        wg.addColorStop(0.18, 'rgba(124,134,224,0.89)')
        wg.addColorStop(0.40, 'rgba(94,106,210,0.87)')
        wg.addColorStop(0.70, 'rgba(60,70,174,0.85)')
        wg.addColorStop(1,    'rgba(24,32,108,0.83)')
        ctx.fillStyle = wg
        ctx.fill()

        // ── Subsurface scatter glow ──
        const sg = ctx.createLinearGradient(0, minY, 0, minY + 70)
        sg.addColorStop(0, 'rgba(200,206,248,0.14)')
        sg.addColorStop(1, 'rgba(94,106,210,0)')
        ctx.fillStyle = sg
        ctx.fill()

        // ── Crest foam (main) ──
        ctx.beginPath()
        ctx.moveTo(pts[0][0], pts[0][1])
        for (const [x, y] of pts) ctx.lineTo(x, y)
        ctx.strokeStyle = 'rgba(255,255,255,0.68)'
        ctx.lineWidth = 3.5
        ctx.lineJoin = 'round'
        ctx.stroke()

        // ── Secondary foam ribbons ──
        for (const [offset, alpha] of [[12, 0.26], [26, 0.14]]) {
          ctx.beginPath()
          for (const [x, y] of pts) ctx.lineTo(x, y + offset)
          ctx.strokeStyle = `rgba(220,224,252,${alpha})`
          ctx.lineWidth = 2
          ctx.stroke()
        }

        // ── Specular highlights on wave face ──
        for (let i = 2; i < pts.length - 2; i += 2) {
          const slope = pts[i + 1][1] - pts[i - 1][1]
          if (slope < -3.5) {
            const s = Math.min((-slope - 3.5) / 7, 1)
            const [x, y] = pts[i]
            const rg = ctx.createRadialGradient(x, y - 4, 0, x, y - 4, 14)
            rg.addColorStop(0, `rgba(255,255,255,${s * 0.75})`)
            rg.addColorStop(1, 'rgba(255,255,255,0)')
            ctx.fillStyle = rg
            ctx.beginPath()
            ctx.arc(x, y - 4, 14, 0, Math.PI * 2)
            ctx.fill()
          }
        }
      }

      // ── Foam particles (gravity + drag) ──
      particles = particles.filter(p => p.life > 0)
      for (const p of particles) {
        p.x  += p.vx * dt
        p.vy += 300 * dt          // gravity
        p.vx *= 1 - 2.5 * dt      // air drag
        p.y  += p.vy * dt
        p.life -= dt
        const a = (p.life / p.maxLife) ** 0.7
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r * a, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(255,255,255,${a * 0.92})`
        ctx.fill()
      }

      raf = requestAnimationFrame(frame)
    }

    let raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [onDone])

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'var(--bg)',
      opacity: out ? 0 : 1,
      transition: out ? 'opacity 400ms ease' : 'none',
      overflow: 'hidden',
    }}>
      <div className="aurora"><div className="blob3"/></div>
      <div className="grain"/>

      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0 }}/>

      {showText && <RevealText/>}

      <style>{`
        @keyframes textReveal {
          from { opacity: 0; transform: translateY(22px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
