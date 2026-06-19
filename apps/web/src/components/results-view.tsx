'use client';
// =============================================================================
// User-facing results view (the "money map").
//
// The refined, non-technical counterpart to the raw dev view in app/page.tsx.
// Normal users see ONLY this; the raw dev view is gated behind the admin login
// (see `isAdmin` in page.tsx). Two modules, as scoped:
//   1. the gross -> take-home tax wedge ("the tax calculations"), and
//   2. an AGGREGATED categorical spend breakdown (topic totals, never the
//      per-category atoms) rendered as the hero donut + a spend bar list.
//
// It reads the analyze response (display_rollup topic totals, tax_breakdown,
// residual_assignment, committed_outflows, savings_waterfall, debt service) and
// presents them — it computes no new model output. Dollar arithmetic mirrors
// the dev view's four-way / tax-wedge derivation so the two never disagree.
//
// Visual identity is inherited from the rest of the app: the warm tan accent
// (#d4a574), Georgia serif for expressive figures, sans for chrome, hairline
// dividers, soft accent-tinted cards, and the existing restrained motion.
// =============================================================================
import { useEffect, useMemo, useState } from 'react';
import Segmented from './segmented';

// Shared font tokens (module scope so child components can use them too).
const SERIF = 'Georgia, "Times New Roman", serif';
const SANS = 'var(--font-geist-sans), Arial, sans-serif';

// ---- Minimal structural type of the analyze response (the subset we read). ---
// Kept local + structural so page.tsx's richer ProfileAnalysis is assignable
// without a shared module; we only name the fields this view consumes.
interface DisplayMember {
  category: string;
  is_pinned: boolean;
  is_balance: boolean;
}
interface DisplayTopic {
  topic: string;
  is_spending: boolean;
  predicted_total: number;
  pinned_total: number;
  members: DisplayMember[];
}
interface CommittedItem {
  label: string;
  annual: number;
  pre_tax?: boolean;
  predicted_topup_annual?: number;
  display_annual?: number;
}
interface Analysis {
  d_variable_annual: number;
  debt_service_annual?: number;
  feasibility_slack: number;
  financial_zone: string;
  display_rollup?: { topics: DisplayTopic[] };
  committed_outflows?: { items: CommittedItem[]; total_annual: number };
  residual_assignment?: {
    savings_investment: { annual: number; framing_state?: string };
    genuine_remainder: { annual: number };
  };
  savings_waterfall?: { fired: boolean; total: number };
  tax_breakdown?: {
    federal_tax: number; state_tax: number; city_tax: number; fica: number;
    state_payroll_tax: number; total_tax: number; take_home: number;
    federal_agi: number; state_eitc: number; effective_rate: number;
    federal_eitc: number; federal_ctc: number;
  };
}

interface ThemeColors {
  background: string; text: string; accent: string;
  buttonBg: string; buttonText: string; border: string;
}

interface ResultsViewProps {
  analysis: Analysis;
  colors: ThemeColors;
  isDarkMode: boolean;
  grossIncomeInput: string;
  name?: string;
  onStartOver: () => void;
  onToggleTheme: () => void;
  isAdmin: boolean;
  onOpenDevView: () => void;
  onExitAdmin: () => void;
}

// Friendly, user-facing labels for the backend topic codes + the synthetic
// allocation buckets (savings/debt/insurance/taxes/pre-tax).
const TOPIC_LABEL: Record<string, string> = {
  food: 'Food',
  housing: 'Housing',
  utilities: 'Utilities',
  transportation_travel: 'Transportation',
  health: 'Health',
  entertainment: 'Entertainment',
  shopping: 'Shopping',
  financial_debt: 'Other',
  savings: 'Savings',
  debt: 'Debt',
  insurance: 'Insurance',
  taxes: 'Taxes',
  pretax: 'Pre-tax benefits',
};

// A cohesive warm-editorial palette anchored on the signature tan (#d4a574):
// analogous tans / ambers / terracotta for spending, with deliberately cooler
// accents (sage for health, teal-slate for savings, slate for taxes, muted
// brick for debt) so the meaningful buckets read apart at a glance. Tuned to
// clear ~4.5:1 separation on BOTH the #0a0a0a dark and #ffffff light canvases.
// Ordered so that segments ADJACENT in the donut (drawn in the order below:
// taxes/pretax then the spending topics then debt/savings/insurance) never sit
// next to a near-twin — warm earth tones alternate with cool slate/sage/teal
// accents and light/dark steps, while the whole set stays in one cohesive
// earthy-editorial family with the tan brand accent reserved for UI chrome.
const SEG_COLOR: Record<string, string> = {
  food: '#c8702f',                 // cinnamon (dark warm)
  housing: '#dcb87f',              // light tan
  utilities: '#6f93a3',            // steel blue (cool)
  transportation_travel: '#b06a41',// terracotta
  health: '#7fa183',               // sage green (cool)
  entertainment: '#e3b75e',        // gold (light)
  shopping: '#9a6e84',             // dusty mauve (cool accent)
  financial_debt: '#8a8079',       // taupe
  debt: '#c25750',                 // brick red
  savings: '#4e8a9e',              // teal (cool)
  insurance: '#b29a76',            // sand
  taxes: '#7c8088',                // slate
  pretax: '#9fb1bb',               // light slate
};
const FALLBACK_COLORS = ['#c8702f', '#6f93a3', '#b06a41', '#7fa183', '#e3b75e', '#9a6e84'];

interface Seg {
  key: string;
  label: string;
  value: number; // annual $
  color: string;
}

// ----------------------------------------------------------------------------
// SVG donut. Segments are explicit stroked arcs (full angular control + gaps),
// not dash-offset hacks, so orientation is unambiguous: 0deg = top, clockwise.
// ----------------------------------------------------------------------------
function polar(cx: number, cy: number, r: number, angleDeg: number): [number, number] {
  const a = ((angleDeg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}
function arcPath(cx: number, cy: number, r: number, startAngle: number, endAngle: number): string {
  const [sx, sy] = polar(cx, cy, r, startAngle);
  const [ex, ey] = polar(cx, cy, r, endAngle);
  const largeArc = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${sx.toFixed(3)} ${sy.toFixed(3)} A ${r} ${r} 0 ${largeArc} 1 ${ex.toFixed(3)} ${ey.toFixed(3)}`;
}

// Size-based slice ordering. The overlap pain is at the vertically CRAMPED poles
// (top/bottom on a wide canvas; the sides on a tall one): many small slices there
// stack their labels into too little room. So we steer the FEW large slices onto
// the cramped axis (sparse → their centred labels have space) and the MANY small
// slices onto the roomy axis (room to fan out). The two largest centre on the two
// poles; each semicircle between them is a size "valley" — large near the poles,
// smallest at the roomy mid-side. Greedy value-balancing of the two arcs keeps the
// second-largest ~opposite the largest. <= 2 slices: just largest-first.
function arrangeBySize(segs: Seg[]): Seg[] {
  if (segs.length <= 2) return [...segs].sort((a, b) => b.value - a.value);
  const sorted = [...segs].sort((a, b) => b.value - a.value);
  const [top, bottom, ...rest] = sorted;            // the two poles
  // Split the remainder into the two arcs, value-balanced (rest is descending →
  // this is LPT scheduling, so the arcs end up ~equal in span → bottom ~opposite).
  const right: Seg[] = [], left: Seg[] = [];
  let rSum = 0, lSum = 0;
  for (const s of rest) {
    if (rSum <= lSum) { right.push(s); rSum += s.value; } else { left.push(s); lSum += s.value; }
  }
  // Within an arc: largest at the two ends (near the poles), smallest in the middle.
  const valley = (arr: Seg[]): Seg[] => {
    const a: Seg[] = [], b: Seg[] = [];
    arr.forEach((s, i) => (i % 2 === 0 ? a : b).push(s));
    return [...a, ...b.reverse()];
  };
  return [top, ...valley(right), bottom, ...valley(left)];
}

// One laid-out label around the ring.
interface LabelItem {
  s: Seg;
  dx: number;     // unit radial direction (for the pinned-lift translate)
  dy: number;
  hw: number;     // half-width of the label block (for collision + ring clearance)
  lx: number;     // label-block centre — sits on the slice's radial ray
  ly: number;
}

function Donut({
  segments, total, activeKey, pinnedKey, onEnter, onLeave, onSegClick,
  centerTop, centerLabel, isDarkMode, fmtMoney, textColor, mutedColor,
}: {
  segments: Seg[];
  total: number;
  activeKey: string | null;   // emphasised = hover ?? pinned
  pinnedKey: string | null;   // the clicked/expanded slice
  onEnter: (k: string) => void;
  onLeave: () => void;
  onSegClick: (k: string, e: { stopPropagation: () => void }) => void;
  centerTop: string;
  centerLabel: string;
  isDarkMode: boolean;
  fmtMoney: (n: number) => string;
  textColor: string;
  mutedColor: string;
}) {
  // Wider than tall: the donut sits in the centre and the side margins are the
  // zones the always-on category labels live in, so they scale with — and stay
  // inside — the chart (never clipped).
  // Thick ring on a moderate overall wheel: r sets the overall size, sw the band
  // width. inner hole = 2*(r - sw/2) stays generous so the centre number keeps a
  // clear margin on both ends.
  // vbH gives the poles vertical breathing room for a 2-line label above/below the
  // ring (the ring itself is unchanged — r/sw fixed — so the donut is the same size;
  // the extra height is label margin, used by the size-based pole labels).
  const vbW = 720, vbH = 408, cx = vbW / 2, cy = vbH / 2, r = 126, sw = 46;
  const outerR = r + sw / 2;
  const explode = 6;        // a pinned slice lifts out just slightly (≈ hover, a touch more)
  // Label typography. The radial clearance scales with the WHEEL (a fraction of
  // the outer radius), not the font, so the margin grows with the donut. TEST
  // style: no leader lines — each label's CENTRE sits on the radial ray through
  // its slice's mid-angle (donut centre → slice centre → label centre colinear),
  // the name stacked over the amount and the whole block centred on that point.
  const LABEL_NAME_SIZE = 14.5;        // px — the category name
  const LABEL_VALUE_SIZE = 13.5;       // px — the dollar amount
  // Base radial gap rim → label centre. Kept SMALL (scales with the wheel) so the
  // pole labels — on the vertically cramped axis — land on-canvas; the clearance
  // pass widens anything that still touches the ring into the roomy axis.
  const labelGap = outerR * 0.20;
  const positive = segments.filter((s) => s.value > 0);
  const gapDeg = positive.length > 1 ? 1.4 : 0;
  const dirOf = (deg: number): [number, number] => {
    const a = ((deg - 90) * Math.PI) / 180;
    return [Math.cos(a), Math.sin(a)];
  };

  // Size-based positioning: steer large slices onto the vertically cramped axis
  // (the poles) and small slices onto the roomy axis. WIDE canvas → poles at
  // top/bottom (0°/180°); TALL canvas → poles at the sides (90°/270°). Only WIDE
  // is reachable today (the viewBox is a fixed 2:1); the tall branch is ready for
  // a future responsive viewBox.
  const wide = vbW >= vbH;
  const poleA = wide ? 0 : 90;          // angle the LARGEST slice is centred on
  const ordered = arrangeBySize(positive);
  const firstSweep = ordered.length ? (ordered[0].value / total) * 360 : 0;
  let cursor = poleA - firstSweep / 2;  // rotate so the largest slice straddles the pole
  const arcs = ordered.map((s) => {
    const sweep = (s.value / total) * 360;
    const rawStart = cursor;
    const start = cursor + gapDeg / 2;
    const end = cursor + sweep - gapDeg / 2;
    cursor += sweep;
    return { s, start: Math.min(start, end - 0.01), end, mid: rawStart + sweep / 2 };
  });

  // ---- Label layout (leader-line-free, radial-centred): each label's CENTRE sits
  // on the radial ray through its slice's mid-angle — donut centre, slice centre,
  // label centre colinear — the block (name over amount) centred there. Two passes
  // keep it readable: vertical de-collision (label↔label) then ring clearance
  // (label↔donut, the hard guarantee). ---
  // Label box metrics. hhB = half-height of the 2-line block; halfW estimates the
  // wider of the two lines (serif advance ≈ charW × font-size × #chars). These let
  // the collision + clearance passes reason about real boxes, not points.
  const hhB = (LABEL_NAME_SIZE + LABEL_VALUE_SIZE) / 2 + 3;
  const charW = 0.56;
  const halfW = (s: Seg) => Math.max(
    s.label.length * LABEL_NAME_SIZE,
    fmtMoney(s.value).length * LABEL_VALUE_SIZE,
  ) * charW / 2;
  const clearance = 10;            // min gap any label box must keep from the ring

  // Place each label's CENTRE on its slice's radial ray, one (small) labelGap out.
  // The pinned slice's whole label is later translated by the same lift vector as
  // its arc (render time), so it stays attached when the slice pops out.
  const items: LabelItem[] = arcs.map((a) => {
    const [dx, dy] = dirOf(a.mid);
    const R = outerR + labelGap;
    return { s: a.s, dx, dy, hw: halfW(a.s), lx: cx + dx * R, ly: cy + dy * R };
  });

  // 1) Vertical de-collision — push apart labels whose boxes overlap BOTH
  //    horizontally and vertically. Space-aware + bidirectional: a colliding pair
  //    separates, and whichever label has MORE open room on its FAR side moves
  //    MORE into it. So a crowded label is never shoved into an even tighter spot,
  //    and a centred pole label (lots of room) absorbs most of the nudge while the
  //    cramped side label barely moves. Each keeps its x; only y is adjusted.
  //    (Gauss–Seidel relaxation — a few passes converge for this handful of boxes.)
  // Spacing scales with the label BLOCK (font-derived), never a fixed px — so the
  // gap stays consistent at any font/wheel size. The empty gap between two stacked
  // labels is a generous fraction of a full block (name over amount).
  const blockH = 2 * hhB;                   // one label block (name + amount)
  const minV = blockH * 1.95;               // centre-to-centre min ≈ a ~0.95-block gap
  const HMARGIN = LABEL_NAME_SIZE * 0.9;    // x-overlap tolerance, font-relative
  const vTop = hhB, vBot = vbH - hhB;       // edges (half a block in), to weigh free space
  for (let pass = 0; pass < 60; pass++) {
    let moved = false;
    for (let i = 0; i < items.length; i++) {
      for (let k = i + 1; k < items.length; k++) {
        const a = items[i], b = items[k];
        if (Math.abs(a.lx - b.lx) >= a.hw + b.hw + HMARGIN) continue;   // not stacked
        const sep = Math.abs(a.ly - b.ly);
        if (sep >= minV) continue;
        const over = minV - sep;
        const up = a.ly <= b.ly ? a : b;    // higher label moves up
        const dn = a.ly <= b.ly ? b : a;    // lower label moves down
        const roomUp = Math.max(1, up.ly - vTop);   // each one's far-side free space
        const roomDn = Math.max(1, vBot - dn.ly);
        up.ly -= over * (roomUp / (roomUp + roomDn));
        dn.ly += over * (roomDn / (roomUp + roomDn));
        moved = true;
      }
    }
    if (!moved) break;
  }

  // 2) Ring clearance — the HARD guarantee a label never overlaps the donut. If a
  //    box still bites the ring, push its centre straight out (away from the donut
  //    centre) until its near edge clears by `clearance`. Pole labels are already
  //    clear (small gap, on the cramped axis); wide side labels get pushed into the
  //    horizontal margin, which a wide canvas has in abundance. Runs LAST so it
  //    has the final say — vertical overflow is fine (the SVG is overflow:visible).
  items.forEach((it) => {
    const ux = it.lx - cx, uy = it.ly - cy;
    const d = Math.hypot(ux, uy) || 1;
    const nx = ux / d, ny = uy / d;
    const reach = it.hw * Math.abs(nx) + hhB * Math.abs(ny);  // box extent toward the centre
    const need = outerR + clearance + reach;
    if (d < need) { it.lx = cx + nx * need; it.ly = cy + ny * need; }
  });

  // Inactive label: the NAME stays prominent, the value clearly recedes.
  const nameInactive = isDarkMode ? 'rgba(255,255,255,0.78)' : 'rgba(0,0,0,0.74)';
  const valInactive = isDarkMode ? 'rgba(255,255,255,0.38)' : 'rgba(0,0,0,0.38)';

  return (
    <svg
      viewBox={`0 0 ${vbW} ${vbH}`}
      width="100%"
      style={{ maxWidth: '100%', display: 'block', overflow: 'visible' }}
      className="donut-enter"
      role="img"
      aria-label={`Income allocation — ${centerLabel}, ${centerTop}`}
    >
      {arcs.map(({ s, start, end, mid }, i) => {
        const emph = activeKey === s.key;
        const dim = activeKey !== null && !emph;
        const [dx, dy] = dirOf(mid);
        const off = s.key === pinnedKey ? explode : 0;
        const tf = off ? `translate(${(dx * off).toFixed(2)} ${(dy * off).toFixed(2)})` : undefined;
        return (
          <path
            key={s.key}
            d={arcPath(cx, cy, r, start, end)}
            transform={tf}
            fill="none"
            stroke={s.color}
            strokeWidth={emph ? sw + 6 : sw}
            strokeLinecap="butt"
            opacity={dim ? 0.4 : 1}
            tabIndex={0}
            role="button"
            aria-label={`${s.label}: ${fmtMoney(s.value)}`}
            style={{
              transition: 'opacity 0.2s ease, stroke-width 0.2s ease, transform 0.2s ease',
              cursor: 'pointer', outline: 'none',
              animation: `donutArcIn 0.6s cubic-bezier(0.4,0,0.2,1) both`,
              animationDelay: `${0.12 + i * 0.05}s`,
            }}
            onMouseEnter={() => onEnter(s.key)}
            onMouseLeave={onLeave}
            onFocus={() => onEnter(s.key)}
            onBlur={onLeave}
            onClick={(e) => onSegClick(s.key, e)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSegClick(s.key, e); } }}
          />
        );
      })}

      {/* Center readout. */}
      <text x={cx} y={cy - 2} textAnchor="middle"
        style={{ fontFamily: SERIF, fontSize: 46, fontWeight: 600, fill: 'currentColor' }}>
        {centerTop}
      </text>
      <text x={cx} y={cy + 30} textAnchor="middle"
        style={{ fontFamily: SERIF, fontSize: 13, letterSpacing: '0.08em', fill: 'currentColor', opacity: 0.55, textTransform: 'uppercase' }}>
        {centerLabel}
      </text>

      {/* Always-on labels (TEST: no leader line/dot, radial-centred): the name
          stacked over the amount, the block centred on the slice's radial ray.
          Faded grey by default, full colour for the active one. */}
      {items.map(({ s, dx, dy, lx, ly }) => {
        const emph = activeKey === s.key;
        // Pinned: translate the whole label by the same lift vector as the arc,
        // so it stays attached and moves out together with the segment.
        const off = s.key === pinnedKey ? explode : 0;
        const tf = off ? `translate(${(dx * off).toFixed(2)} ${(dy * off).toFixed(2)})` : undefined;
        return (
          <g key={s.key} transform={tf} style={{ pointerEvents: 'none', transition: 'transform 0.2s ease' }}>
            <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" style={{ fontFamily: SERIF }}>
              <tspan x={lx} dy="-0.55em" style={{ fontSize: LABEL_NAME_SIZE, fontWeight: emph ? 600 : 500, fill: emph ? textColor : nameInactive, transition: 'fill 0.2s ease' }}>
                {s.label}
              </tspan>
              <tspan x={lx} dy="1.25em" style={{ fontSize: LABEL_VALUE_SIZE, fontWeight: 500, fill: emph ? mutedColor : valInactive, transition: 'fill 0.2s ease' }}>
                {fmtMoney(s.value)}
              </tspan>
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function ResultsView({
  analysis, colors, isDarkMode, grossIncomeInput,
  onStartOver, onToggleTheme, isAdmin, onOpenDevView, onExitAdmin,
}: ResultsViewProps) {
  const [unit, setUnit] = useState<'monthly' | 'annual'>('monthly');
  // The donut shows the take-home allocation (the gross view was dropped); the
  // union type is kept so the gross-aware arithmetic below still type-checks.
  const [mode] = useState<'take_home' | 'gross'>('take_home');
  // Hover = transient preview; pinned = click-locked until the user clicks
  // elsewhere or opens another section. Emphasised slice = hover ?? pinned.
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const [pinnedKey, setPinnedKey] = useState<string | null>(null);
  const [showTaxDetail, setShowTaxDetail] = useState(false);
  const emphasizedKey = hoverKey ?? pinnedKey;
  const pinSlice = (k: string, e: { stopPropagation: () => void }) => {
    e.stopPropagation();   // keep the document-level clear (below) from firing for this click
    setPinnedKey((cur) => (cur === k ? null : k));
  };
  // A click anywhere outside a slice — or opening another section, which also
  // bubbles to document — releases the pin.
  useEffect(() => {
    if (pinnedKey === null) return;
    const clear = () => setPinnedKey(null);
    document.addEventListener('click', clear);
    return () => document.removeEventListener('click', clear);
  }, [pinnedKey]);

  const div = unit === 'monthly' ? 12 : 1;
  const fmt = (n: number) =>
    Math.round(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
  const money = (n: number) => `$${fmt(n / div)}`;
  // A deduction line: negative tax (refundable EITC/CTC exceeding liability)
  // reads as a credit, not a confusing "– $-200" double negative.
  const deduction = (n: number) => (n < 0 ? `+ ${money(Math.abs(n))}` : `– ${money(n)}`);
  const unitWord = unit === 'monthly' ? '/mo' : '/yr';

  // ---- Derived dollars (mirrors the dev view's four-way / tax-wedge math) ----
  const derived = useMemo(() => {
    const tb = analysis.tax_breakdown;
    const committed = analysis.committed_outflows?.total_annual ?? 0;
    const grossAnnual = Number.isFinite(parseFloat(grossIncomeInput))
      ? parseFloat(grossIncomeInput)
      : (tb ? tb.take_home + tb.total_tax : 0);
    const pretaxContrib = tb ? Math.max(0, grossAnnual - tb.federal_agi) : 0;
    const takeHome = tb ? tb.take_home - pretaxContrib : analysis.d_variable_annual + committed;

    const items = (analysis.committed_outflows?.items ?? []).filter((it) => it.annual > 0);
    const itTopup = items.reduce((s, it) => s + (it.predicted_topup_annual ?? 0), 0);
    const postTaxCommitted = items
      .filter((it) => !it.pre_tax)
      .reduce((s, it) => s + (it.display_annual ?? it.annual), 0);

    const waterfall = analysis.savings_waterfall?.total ?? 0;
    // No Math.max guard — mirrors the dev view's `waterfall - itTopup` exactly
    // (itTopup is a strict subset of the routed total, so this is always >= 0).
    const postTaxWaterfall = waterfall - itTopup;
    const savingsLine = analysis.residual_assignment?.savings_investment.annual ?? 0;
    const remainder = analysis.residual_assignment?.genuine_remainder.annual ?? 0;
    const savings = savingsLine + postTaxWaterfall + remainder;

    const debt = analysis.debt_service_annual ?? 0;

    // Spending topics (aggregated). predicted + pinned per spending topic.
    const topics = (analysis.display_rollup?.topics ?? [])
      .filter((t) => t.is_spending)
      .map((t, i) => ({
        key: t.topic,
        label: TOPIC_LABEL[t.topic] ?? t.topic,
        value: t.predicted_total + t.pinned_total,
        color: SEG_COLOR[t.topic] ?? FALLBACK_COLORS[i % FALLBACK_COLORS.length],
      }))
      .filter((t) => t.value > 0.5);
    const spendingTotal = topics.reduce((s, t) => s + t.value, 0);

    const taxLines = tb
      ? tb.federal_tax + tb.fica + tb.state_tax + tb.state_payroll_tax + tb.city_tax
      : 0;

    return {
      tb, grossAnnual, pretaxContrib, takeHome, postTaxCommitted, savings, debt,
      topics, spendingTotal, taxLines, items,
    };
  }, [analysis, grossIncomeInput]);

  // ---- Donut segments for the active mode ------------------------------------
  const segments: Seg[] = useMemo(() => {
    const spend: Seg[] = derived.topics.map((t) => ({
      key: t.key, label: t.label, value: t.value, color: t.color,
    }));
    const tail: Seg[] = [];
    if (derived.debt > 0.5)
      tail.push({ key: 'debt', label: TOPIC_LABEL.debt, value: derived.debt, color: SEG_COLOR.debt });
    if (derived.savings > 0.5)
      tail.push({ key: 'savings', label: TOPIC_LABEL.savings, value: derived.savings, color: SEG_COLOR.savings });

    const takeHomeSegs = [...spend, ...tail];
    if (mode === 'take_home') return takeHomeSegs;

    // Gross mode: prepend taxes + pre-tax benefits so the donut sums to gross.
    // A net refund can drive taxLines <= 0; clamp the arc to >= 0 (a refund
    // can't be drawn as a wedge) — the headline figure is still the true gross.
    const head: Seg[] = [];
    if (derived.taxLines > 0.5)
      head.push({ key: 'taxes', label: TOPIC_LABEL.taxes, value: derived.taxLines, color: SEG_COLOR.taxes });
    if (derived.pretaxContrib > 0.5)
      head.push({ key: 'pretax', label: TOPIC_LABEL.pretax, value: derived.pretaxContrib, color: SEG_COLOR.pretax });
    return [...head, ...takeHomeSegs];
  }, [derived, mode]);

  const donutTotal = segments.reduce((s, x) => s + x.value, 0) || 1;
  // The center HEADLINE is always the mode's authoritative figure (true gross /
  // true take-home), never the raw segment sum — so a deficit (cohort spend >
  // budget) or a refund edge can never make the center silently overshoot.
  const headlineTotal = mode === 'gross' ? derived.grossAnnual : derived.takeHome;
  // The centre always holds the headline figure; the hovered slice's own label
  // is shown on the OUTSIDE of the ring (the donut's leader-line callout).
  const centerTop = money(headlineTotal);
  // No unit suffix inside the donut — the Monthly/Annual control conveys it.
  const centerLabel = mode === 'gross' ? 'gross' : 'take-home';

  // ---- shared style tokens ---------------------------------------------------
  const hairline = isDarkMode ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.10)';
  const muted = isDarkMode ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.50)';
  const cardBg = isDarkMode ? 'rgba(212,165,116,0.07)' : 'rgba(212,165,116,0.06)';
  // Accent-as-TEXT token: the tan #d4a574 fails AA on white (2.2:1), so light
  // mode uses a darker tan for accent figures (the hero take-home number).
  const accentText = isDarkMode ? colors.accent : '#9a6f3d';
  const serif = SERIF;
  const sans = SANS;
  const wedgeBase = { colors, serif, sans, hairline, muted, accentText };

  const tb = derived.tb;

  return (
    <div style={{
      minHeight: '100vh', background: colors.background, color: colors.text,
      fontFamily: serif, padding: '3vh 5vw 9vh',
    }}>
      <style>{`
        @keyframes donutArcIn { from { opacity: 0; } to { opacity: 1; } }
        .donut-enter { animation: donutPop 0.6s cubic-bezier(0.34,1.56,0.64,1) both; }
        @keyframes donutPop { from { opacity: 0; transform: scale(0.92); } to { opacity: 1; transform: scale(1); } }
        .rv-rise { animation: rvRise 0.55s cubic-bezier(0.4,0,0.2,1) both; }
        @keyframes rvRise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        /* On narrow screens the Start-over button collapses to just the arrow. */
        @media (max-width: 640px) {
          .rv-so-text { display: none; }
          .rv-so { padding: 9px 12px !important; }
        }
      `}</style>

      <div style={{ maxWidth: 1080, margin: '0 auto' }}>
        {/* Start over — pinned to the viewport's top-left corner regardless of
            window width (like the home page hamburger), out of the flow. */}
        <button
          onClick={onStartOver}
          className="rv-so"
          style={{
            position: 'fixed', top: 26, left: 22, zIndex: 1000,
            display: 'inline-flex', alignItems: 'center', gap: 11, backgroundColor: colors.background,
            border: `1px solid ${hairline}`, color: colors.text, cursor: 'pointer',
            borderRadius: 9999, padding: '8px 18px', fontFamily: sans, fontSize: 13,
            transition: 'border-color 0.25s ease, color 0.25s ease',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = colors.accent; e.currentTarget.style.color = colors.accent; }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = hairline; e.currentTarget.style.color = colors.text; }}
        >
          <span aria-hidden="true">←</span>
          <span className="rv-so-text">Start over</span>
        </button>

        {/* ---- Top bar: Monthly/Annual centered; admin pinned to the right.
             (Start over is the fixed top-left button above.) ---- */}
        <div style={{
          position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center',
          minHeight: 40, marginTop: '1.6vh', marginBottom: '2vh',
        }}>
          <Segmented
            value={unit}
            onChange={setUnit}
            options={[{ v: 'monthly', label: 'Monthly' }, { v: 'annual', label: 'Annual' }]}
            hairline={hairline} accent={colors.accent} text={colors.text}
          />
          {isAdmin && (
            <div style={{
              position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)',
              display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
            }}>
              <span style={{
                fontFamily: sans, fontSize: 11, letterSpacing: '0.14em', textTransform: 'uppercase',
                color: colors.accent, border: `1px solid ${colors.accent}`, borderRadius: 9999,
                padding: '4px 10px',
              }}>
                admin
              </span>
              <button
                onClick={onOpenDevView}
                style={{
                  background: 'transparent', border: `1px solid ${hairline}`, color: colors.text,
                  cursor: 'pointer', borderRadius: 9999, padding: '8px 16px', fontFamily: sans, fontSize: 13,
                }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = colors.accent; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = hairline; }}
              >
                Developer view
              </button>
              <button
                onClick={onExitAdmin}
                title="Sign out of admin"
                style={{
                  background: 'transparent', border: 'none', color: muted, cursor: 'pointer',
                  fontFamily: sans, fontSize: 13,
                }}
              >
                exit
              </button>
            </div>
          )}
        </div>

        {/* ---- Hero: the donut fills the first screen. Click a slice to pin
             it (it pops out + its label lights up) until you click elsewhere. */}
        <div className="rv-rise" style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: 22, minHeight: 'calc(100vh - 96px)', animationDelay: '0.05s',
        }}>
          <div style={{ width: '100%', maxWidth: 'min(1040px, 96vw)', display: 'flex', justifyContent: 'center' }}>
            <Donut
              segments={segments}
              total={donutTotal}
              activeKey={emphasizedKey}
              pinnedKey={pinnedKey}
              onEnter={setHoverKey}
              onLeave={() => setHoverKey(null)}
              onSegClick={pinSlice}
              centerTop={centerTop}
              centerLabel={centerLabel}
              isDarkMode={isDarkMode}
              fmtMoney={money}
              textColor={colors.text}
              mutedColor={muted}
            />
          </div>
        </div>

        {/* ---- Tax wedge card ---- */}
        {tb && (
          <section className="rv-rise" style={{ marginTop: '6vh', animationDelay: '0.1s' }}>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: '2.4vh' }}>
              <h2 style={{ fontFamily: serif, fontWeight: 400, fontSize: '1.7rem', margin: 0, letterSpacing: '-0.01em' }}>
                Estimated Taxes
              </h2>
              <span style={{ fontFamily: sans, fontSize: 13, color: muted }}>
                what taxes &amp; pre-tax deductions take out
              </span>
            </div>

            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16,
              marginBottom: '3vh',
            }}>
              {[
                { label: 'Gross income', value: money(derived.grossAnnual), big: true },
                {
                  label: tb.total_tax < 0 ? 'Net tax refund' : 'Total tax',
                  value: money(Math.abs(tb.total_tax)),
                  sub: `${(tb.effective_rate * 100).toFixed(1)}% effective`,
                },
                { label: 'Take-home', value: money(derived.takeHome), accent: true },
              ].map((stat) => (
                <div key={stat.label} style={{
                  background: cardBg, border: `1px solid ${hairline}`, borderRadius: 14,
                  padding: '2.4vh 1.6vw',
                }}>
                  <div style={{ fontFamily: serif, fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase', color: muted, marginBottom: 8 }}>
                    {stat.label}
                  </div>
                  <div style={{ fontFamily: serif, fontSize: stat.big ? '2.2rem' : '1.9rem', fontWeight: 600, color: stat.accent ? accentText : colors.text }}>
                    {stat.value}
                    <span style={{ fontFamily: sans, fontSize: 13, fontWeight: 400, color: muted, marginLeft: 6 }}>{unitWord}</span>
                  </div>
                  {stat.sub && (
                    <div style={{ fontFamily: sans, fontSize: 13, color: muted, marginTop: 4 }}>{stat.sub}</div>
                  )}
                </div>
              ))}
            </div>

            {/* Line-by-line wedge */}
            <div style={{ border: `1px solid ${hairline}`, borderRadius: 14, overflow: 'hidden' }}>
              <WedgeRow {...wedgeBase} label="Gross income" value={money(derived.grossAnnual)} strong first />
              <WedgeRow {...wedgeBase} label="Federal income tax" value={deduction(tb.federal_tax)}
                note={tb.federal_tax < 0 ? 'net refund — refundable credits' : (tb.federal_eitc > 0 || tb.federal_ctc > 0 ? 'after credits' : undefined)} />
              <WedgeRow {...wedgeBase} label="FICA — Social Security & Medicare" value={deduction(tb.fica)} />
              <WedgeRow {...wedgeBase} label="State income tax" value={deduction(tb.state_tax)}
                note={tb.state_eitc > 0 ? `net of state EITC ${money(tb.state_eitc)}` : undefined} />
              {tb.state_payroll_tax > 0 && (
                <WedgeRow {...wedgeBase} label="State payroll (disability / paid leave)" value={deduction(tb.state_payroll_tax)} />
              )}
              {tb.city_tax > 0 && (
                <WedgeRow {...wedgeBase} label="Local / municipal tax" value={deduction(tb.city_tax)} />
              )}
              {derived.pretaxContrib > 0.5 && (
                <button
                  type="button"
                  aria-expanded={showTaxDetail}
                  onClick={() => setShowTaxDetail((v) => !v)}
                  style={{
                    display: 'grid', gridTemplateColumns: '1fr auto', gap: 12, alignItems: 'center',
                    width: '100%', textAlign: 'left', padding: '1.5vh 1.4vw',
                    // Top-only border via longhands only — never mix the `border`
                    // shorthand with `borderTop*`, or React warns when hairline
                    // changes value on a re-render (e.g. dark/light toggle).
                    borderWidth: '1px 0 0', borderStyle: 'solid', borderColor: hairline,
                    cursor: 'pointer', background: 'transparent',
                    color: colors.text, font: 'inherit',
                  }}
                >
                  <span style={{ fontFamily: serif, fontSize: 15, display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ color: muted, fontSize: 11 }}>{showTaxDetail ? '▾' : '▸'}</span>
                    Pre-tax benefits &amp; retirement
                    <span style={{ fontFamily: sans, color: muted, fontSize: 12 }}>(reduce taxable income)</span>
                  </span>
                  <span style={{ fontFamily: serif, fontSize: 16, fontWeight: 600 }}>– {money(derived.pretaxContrib)}</span>
                </button>
              )}
              {showTaxDetail && derived.items.filter((it) => it.pre_tax).map((it) => (
                <div key={it.label} style={{
                  display: 'grid', gridTemplateColumns: '1fr auto', gap: 12,
                  padding: '1vh 1.4vw 1vh 2.8vw', background: cardBg,
                }}>
                  <span style={{ fontFamily: serif, fontSize: 13.5, color: muted }}>
                    {it.label.replace(/\s*\([^)]*\)\s*$/, '').replace(/\s*[—–]\s*employee share\s*$/i, '')}
                  </span>
                  <span style={{ fontFamily: serif, fontSize: 14, color: muted }}>{money(it.display_annual ?? it.annual)}</span>
                </div>
              ))}
              <WedgeRow {...wedgeBase} label="Take-home pay" value={money(derived.takeHome)} strong accent />
            </div>
          </section>
        )}

        {/* Closing note (predict-not-prescribe). */}
        <p style={{ fontFamily: sans, fontSize: 12.5, color: muted, marginTop: '5vh', lineHeight: 1.6, maxWidth: 720 }}>
          These are cohort-typical estimates for a household like yours in your area — a starting
          picture to adjust, not a budget set in stone. Figures are rounded.
        </p>
      </div>

      {/* Theme toggle (kept consistent with the rest of the app) */}
      <div style={{ position: 'fixed', bottom: 15, right: 15, zIndex: 1500 }}>
        <button
          onClick={onToggleTheme}
          aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-pressed={isDarkMode}
          title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          style={{
            width: 32, height: 16, borderRadius: 8, border: 'none', background: colors.accent,
            cursor: 'pointer', position: 'relative', outline: 'none',
          }}
        >
          <span style={{
            width: 12, height: 12, borderRadius: '50%', background: isDarkMode ? '#000' : '#fff',
            position: 'absolute', top: 2, left: isDarkMode ? 2 : 18, transition: 'left 0.3s ease',
            boxShadow: '0 1px 2px rgba(0,0,0,0.3)',
          }} />
        </button>
      </div>
    </div>
  );
}

function WedgeRow({
  label, value, note, strong, accent, first, colors, serif, sans, hairline, muted, accentText,
}: {
  label: string; value: string; note?: string; strong?: boolean; accent?: boolean; first?: boolean;
  colors: ThemeColors; serif: string; sans: string; hairline: string; muted: string; accentText: string;
}) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '1fr auto', gap: 12, alignItems: 'center',
      padding: '1.5vh 1.4vw',
      borderTop: first ? 'none' : `1px solid ${hairline}`,
    }}>
      <span style={{ fontFamily: serif, fontSize: strong ? 16 : 15, fontWeight: strong ? 600 : 400, color: colors.text }}>
        {label}
        {note && <span style={{ fontFamily: sans, color: muted, fontSize: 12.5, marginLeft: 8 }}>{note}</span>}
      </span>
      <span style={{
        fontFamily: serif, fontSize: strong ? 19 : 16, fontWeight: 600,
        color: accent ? accentText : colors.text,
      }}>
        {value}
      </span>
    </div>
  );
}
