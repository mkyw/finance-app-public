'use client';
// A pill segmented control with a SLIDING active thumb. Shared by the results
// view (Monthly | Annual) and the landing form (Individual | Family). Module
// scope + stable identity so it never remounts mid-interaction.
import { CSSProperties } from 'react';

const SANS = 'var(--font-geist-sans), Arial, sans-serif';

export default function Segmented<T extends string>({
  value, options, onChange, accent, text, hairline,
  width, optionWidth = 92, height = 34, fontSize = 13, activeColor = '#1a1207',
}: {
  value: T | null;
  options: { v: T; label: string }[];
  onChange: (v: T) => void;
  accent: string;             // sliding thumb colour
  text: string;               // inactive label colour
  hairline: string;           // border colour
  width?: number;             // total fixed width (e.g. 240); else N×optionWidth
  optionWidth?: number;       // per-option width when `width` is omitted
  height?: number;
  fontSize?: number;
  activeColor?: string;       // label colour over the thumb
}) {
  const n = options.length;
  const border = 1, pad = 3;
  const totalW = width ?? n * optionWidth + 2 * (border + pad);
  const innerW = totalW - 2 * (border + pad);
  const innerH = height - 2 * (border + pad);
  const thumbW = innerW / n;
  const idx = value == null ? -1 : options.findIndex((o) => o.v === value);

  const container: CSSProperties = {
    position: 'relative', display: 'inline-flex', boxSizing: 'border-box',
    width: totalW, height, padding: pad,
    border: `${border}px solid ${hairline}`, borderRadius: 9999, background: 'transparent',
  };
  const thumb: CSSProperties = {
    position: 'absolute', top: pad, left: pad, width: thumbW, height: innerH,
    borderRadius: 9999, background: accent, zIndex: 0,
    transform: `translateX(${Math.max(0, idx) * thumbW}px)`,
    opacity: idx < 0 ? 0 : 1,
    transition: 'transform 0.28s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.2s ease',
  };

  return (
    <div style={container}>
      <div style={thumb} aria-hidden="true" />
      {options.map((o) => {
        const on = o.v === value;
        return (
          <button
            key={o.v}
            type="button"
            aria-pressed={on}
            onClick={() => onChange(o.v)}
            style={{
              flex: '1 1 0', position: 'relative', zIndex: 1,
              border: 'none', background: 'transparent', cursor: 'pointer',
              borderRadius: 9999, padding: 0, textAlign: 'center',
              fontFamily: SANS, fontSize, fontWeight: on ? 600 : 500, letterSpacing: '0.01em',
              color: on ? activeColor : text, transition: 'color 0.25s ease', outline: 'none',
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
