import { cn } from '@/lib/utils'
import type { Model } from '@/lib/types'

const MODELS: Model[] = ['WS Gap', 'Monster', 'Aqua', 'Pure', 'Gem', 'Gem v2']

interface Props {
  active: Model
  onChange: (m: Model) => void
}

export function ModelTabs({ active, onChange }: Props) {
  return (
    <div className="flex gap-[5px] items-center" role="tablist" aria-label="AI Model">
      {MODELS.map((m) => (
        <button
          key={m}
          role="tab"
          aria-selected={active === m}
          onClick={() => onChange(m)}
          className={cn(
            // base
            'font-sans text-[11px] font-semibold cursor-pointer whitespace-nowrap select-none',
            'px-[13px] py-[7px] rounded-full',
            'transition-all duration-200 ease-spring',
            'min-h-[36px] min-w-[44px]', // touch target
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-capper-indigo/60',
            // state
            active === m
              ? 'pill-glass-active text-white scale-[1.04]'
              : 'pill-glass text-white/40 hover:text-white/65',
            'active:scale-[0.96]',
          )}
        >
          {m}
        </button>
      ))}
    </div>
  )
}
