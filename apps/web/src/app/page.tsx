'use client'
import Image from "next/image";
import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import RotatingText from '../components/rotating-text';
import ResultsView from '../components/results-view';
import Segmented from '../components/segmented';
import { getColors, getButtonStyles, layoutStyles, getToggleStyles } from '../styles/theme';
import { initCityList, searchCities, type CityOption } from '../../lib/city-search';
import { resolveCity, type ResolvedCity } from '../../lib/resolve-city';

interface BenefitMatch {
  program_name: string;
  estimated_monthly_min: number;
  estimated_monthly_max: number;
  confidence: string;
  enrollment_url: string;
}

interface SpendingDistribution {
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  engel_estimate: number;
  feasibility_adjusted: number;
  cohort_position: number;
  is_structural: boolean;
  behavioral_gap: number;
  nonzero_rate: number;
  conditional_p90: number;
  // Build 4 / 6 back-fill split — measured (feasibility_adjusted) is the
  // cohort-typical baseline; backfill_inferred is the post-allocation
  // discretionary-accounting increment (≥ 0; nonzero only when the back-fill
  // fires for a slope-ceiling target category). Displayed value of the line
  // = feasibility_adjusted + backfill_inferred; both must be summed into any
  // group total or the four-way accounting won't reconcile. Optional ?? 0
  // for backward compat with pre-Build-4 backend responses.
  backfill_inferred?: number;
  backfill_confidence?: string;
}

// ---------------------------------------------------------------------------
// The topic grouping is NOT defined here. It comes from the analyze API's
// `display_rollup` (backend apps/api/profiles/display.py, derived from
// shared/constants/categories.py) — a single source of truth. The view renders
// whatever grouping/cadence the backend sends; there is deliberately no local
// mirror to drift. Episodic lines (atom.cadence === "episodic") get CSS italics
// per row at render time — never ANSI escape codes.

// Topic roll-up sent by the analyze API (apps/api/profiles/display.py).
interface DisplayMember {
  category: string;
  value: number;
  cadence: string;              // "recurring" | "episodic" | "balance"
  is_pinned: boolean;
  is_balance: boolean;
  episodic_subcomponents: string[];
  // Heavy-zero, cohort-mean-meaningless cats (chrty/educ/ocash/stdint/othint/
  // finpay) — value-layer-zeroed; omitted from the initial view.
  omit_from_initial_view?: boolean;
}
interface DisplayTopic {
  topic: string;
  is_spending: boolean;
  predicted_total: number;
  pinned_total: number;
  members: DisplayMember[];
}
interface DisplayRollup {
  topics: DisplayTopic[];
}

interface CommittedOutflowItem {
  code: string;
  label: string;
  annual: number;
  monthly: number;
  source: string;
  adjustable: boolean;
  // Up-direction waterfall top-up fields (7c3, 2026-06-10). 0 when waterfall
  // didn't fire or didn't reach this line. `display_annual` = annual + topup.
  predicted_topup_annual?: number;
  predicted_topup_monthly?: number;
  display_annual?: number;
  // Display-partition flag: pre-tax outflows (reduce AGI) render in the tax wedge
  // between taxes and take-home; post-tax ones stay in the four-way.
  pre_tax?: boolean;
}
interface CommittedOutflowsBlock {
  items: CommittedOutflowItem[];
  total_annual: number;
  // `total_with_topups_annual` is the displayed total (allocation + topups);
  // `total_annual` is the d_var input (unchanged — no circularity).
  total_with_topups_annual?: number;
}
interface WaterfallFill {
  code: string;
  label: string;
  annual: number;
  monthly: number;
  current: number;
  limit: number;
  headroom: number;
  maxed: boolean;
  mechanism: string;
  adjustable: boolean;
}
interface SavingsWaterfallBlock {
  fired: boolean;
  trigger: string;
  total: number;
  limits_year: number;
  fills: WaterfallFill[];
}
interface ResidualSweepBlock {
  fired: boolean;
  trigger: string;
  total: number;
  swept: Record<string, number>;
}
interface ResidualLine {
  annual: number;
  monthly: number;
  label: string;
  source?: string;
  adjustable?: boolean;
  // Savings line only: drives state-dependent display copy.
  framing_state?:
    | "signal_confirmed_cohort"
    | "signal_pulled_down"
    | "signal_would_pull_up_deferred"
    | "signal_pulled_up_routed"
    | "user_pinned";
}
interface ResidualAssignmentBlock {
  savings_investment: ResidualLine;
  genuine_remainder: ResidualLine;
  realistic_savings_rate: number;
  realistic_savings_cap: number;
}
interface BackfillAudit {
  fired: boolean;
  s_star: number;
  // Personalized (blended) benchmark the trigger/pool used (spend-arm build):
  // == s_star without a balance signal; below it when a low reported balance
  // pulled the back-fill benchmark down (larger discretionary lift).
  s_star_personalized?: number;
  residual_rate: number;
  g: number;
  pool: number;
  // Per-category inferred dollars (the back-fill lift on each slope-ceiling
  // discretionary target — entertainment, shopping, hotel, airshp, recrp,
  // eatout). Only categories with inferred > 0 appear.
  inferred: Record<string, number>;
  feasibility_slack_pre_backfill: number;
  // High-earner ceiling stratification (2026-06-09): the matched pool's
  // median y_eq (the threshold) + whether this profile read the
  // high-earner-subset conditional_p90_hi caps instead of broad cp90.
  cohort_median_y_eq?: number;
  stratified?: boolean;
}
// Debt-accumulation annotation (soft-deficit-with-CC-debt regime, 2026-06-09).
// Conditional and OUTSIDE the four-way sum — the compressed allocation already
// sums to the post-debt budget; this is the counterfactual "cohort-typical
// spending runs $X above it, which could add to your card balance". Fires only
// on solver_status=="soft_constrained" + a reported carried CC balance;
// applies:false on every other path. framing_state drives the copy.
interface DebtAccumulationBlock {
  applies: boolean;
  monthly_potential_growth?: number;
  annual_potential_growth?: number;
  basis?: string;
  source?: string;
  framing_state?: 'signal_clear' | 'signal_marginal' | 'user_pinned';
  adjustable?: boolean;
  cc_balance_to_income?: number;
  gap_ratio?: number;
}

interface ProfileAnalysis {
  financial_zone: string;
  structural_deficit: number;
  feasibility_slack: number;
  d_variable_annual: number;
  d_variable_adjusted?: number;
  debt_service_annual?: number;
  pace_annual: number;
  solver_status: string;
  distributions: Record<string, SpendingDistribution>;
  display_rollup: DisplayRollup;
  // Build-5/6 blocks — present on aggregated-path responses (the default).
  committed_outflows?: CommittedOutflowsBlock;
  residual_assignment?: ResidualAssignmentBlock;
  backfill?: BackfillAudit;
  debt_accumulation?: DebtAccumulationBlock;
  // Down-direction residual sweep (2026-06-10): swept dollars surface in
  // distributions as backfill_inferred; this block is the audit split.
  residual_sweep?: ResidualSweepBlock;
  // Up-direction savings waterfall (2026-06-10): fills are rendered as their
  // own section; total enters the four-way sum (not committed, no double-count).
  savings_waterfall?: SavingsWaterfallBlock;
  // Per-debt-type liabilities (debt-input build). Keyed by othdbt / stddbt /
  // auto_loan / other_debt. `source` ∈ user_reported | cohort_predicted |
  // not_modeled; not-modeled $0 components are omitted from display.
  balance_sheet?: {
    liabilities?: Record<string, {
      label: string;
      annual_service: number;
      monthly_service: number;
      source: string;
      adjustable?: boolean;
      predicted_balance?: number;
      reported_balance?: number;
      reported_monthly_payment?: number;
      // True for cohort_predicted + not_modeled lines (omitted from initial
      // view); False only for user_reported debt (the lines we surface).
      omit_from_initial_view?: boolean;
    }>;
    total_debt_service_annual?: number;
  };
  benefits: BenefitMatch[];
  match_metadata: {
    n_effective: number;
    confidence: string;
    pumas_used: string[];
    city_pumas_used: string[];
    n_households: number;
    car_owner_classification: 'owner' | 'non_owner' | 'ambiguous';
    car_owner_probability: number;
  };
  // Stage 6: the gross->take-home tax wedge (oracle-verified via Gate 1) +
  // the imputed filing unit (surfaced so the user can correct it).
  tax_breakdown?: {
    federal_tax: number; state_tax: number; city_tax: number; fica: number;
    state_payroll_tax: number; total_tax: number; take_home: number; tax_year: number;
    federal_amt: number; federal_niit: number; federal_eitc: number; federal_ctc: number;
    federal_agi: number; federal_taxable_income: number;
    itemized_deductions: number; standard_deduction: number; itemized: boolean;
    state_eitc: number; effective_rate: number;
  };
  filing_unit?: {
    filing_status: string; declared_filing_status: string; num_dependents: number;
    n_children_under_18: number; n_children_under_17: number; n_children_under_13: number;
    n_children_under_6: number; eic_qualifying_children: number;
    dependent_ages: number[]; spouse_age: number | null; imputed: boolean; notes: string[];
  };
}

// Screen-2 debt field: a $-prefixed inline-styled input matching the
// conversational money inputs on Screen 1. Defined at MODULE scope (not inside
// Home) so it keeps a stable component identity across re-renders — a function
// component re-declared inside render remounts on every keystroke and drops
// input focus. `disabled` fades it (skip-all).
function DollarInput({
  value, onChange, placeholder, width = 120, disabled = false, accent, text,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  width?: number;
  disabled?: boolean;
  accent: string;
  text: string;
}) {
  return (
    <span style={{ position: 'relative', display: 'inline-block', opacity: disabled ? 0.4 : 1 }}>
      <span style={{
        position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)',
        color: text, fontSize: '1.25rem',
        fontFamily: 'Georgia, "Times New Roman", serif', fontWeight: '600',
        pointerEvents: 'none', zIndex: 1,
      }}>$</span>
      <input
        type="number"
        min="0"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: `${width}px`,
          padding: '0.125vh 0.5vw 0.25vh 20px',
          borderWidth: '0 0 2px',
          borderStyle: 'solid',
          borderColor: accent,
          backgroundColor: 'transparent',
          color: accent,
          fontWeight: '600',
          outline: 'none',
          textAlign: 'center',
          fontSize: '1.25rem',
          fontFamily: 'Georgia, "Times New Roman", serif',
        }}
        placeholder={placeholder}
      />
    </span>
  );
}

export default function Home() {
  const router = useRouter();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isThemeLoaded, setIsThemeLoaded] = useState(false);

  // Form states from start page
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [income, setIncome] = useState('');
  const [savings, setSavings] = useState('');
  const [householdSize, setHouseholdSize] = useState('');
  const [tenure, setTenure] = useState<'OWN' | 'RENT' | null>(null);
  const [housingCost, setHousingCost] = useState('');
  // Two-screen flow: 1 = personal info (existing), 2 = debt collection.
  // State lives here in the parent so Back/Next never resets either screen.
  const [currentScreen, setCurrentScreen] = useState<1 | 2>(1);
  // Screen-2 debt inputs (strings for controlled inputs; '' → 0 on submit,
  // which the backend treats as "use cohort prior"). skipAllDebt fast-paths
  // no-debt users — when checked, all four submit as 0 regardless of any
  // stale typed values (the user's explicit "no debt" statement wins).
  const [ccCarriedBalance, setCcCarriedBalance] = useState('');
  const [studentLoanPayment, setStudentLoanPayment] = useState('');
  const [autoLoanPayment, setAutoLoanPayment] = useState('');
  const [otherDebtPayment, setOtherDebtPayment] = useState('');
  const [skipAllDebt, setSkipAllDebt] = useState(false);
  const [ccTooltipOpen, setCcTooltipOpen] = useState(false);
  const [analysis, setAnalysis] = useState<ProfileAnalysis | null>(null);
  const [showResults, setShowResults] = useState(false);
  // Admin gating. The raw dev view is no longer the default result — normal
  // users see the polished ResultsView; the dev view is reachable only when an
  // admin is signed in (via the Log In modal) and has toggled into it. This is
  // a CLIENT-SIDE demo gate (no backend auth exists yet — the login flow is
  // still a stub), persisted to localStorage so a refresh keeps admin state.
  // Demo credentials: admin@finance.app / devadmin (see handleLogin).
  const [isAdmin, setIsAdmin] = useState(false);
  const [devView, setDevView] = useState(false);
  // Loading view: shown while POST /analyze is in flight. loadingPhase indexes
  // LOADING_PHASES; a timer advances it and a minimum display window keeps the
  // staged copy readable even when the backend answers in well under a second.
  const [isLoading, setIsLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState(0);
  const LOADING_PHASES = [
    'Analyzing your inputs',
    'Matching with similar households',
    'Building your spending profile',
    'Refining your plan',
  ];
  // Dev-view: unit toggle (monthly = value/12) and per-topic-group collapse
  // state (a group absent from the map is collapsed — collapsed by default).
  const [devUnit, setDevUnit] = useState<'monthly' | 'annual'>('monthly');
  const [devCollapsed, setDevCollapsed] = useState<Record<string, boolean>>({});
  const [submitError, setSubmitError] = useState<string>('');
  const [errors, setErrors] = useState({
    name: false,
    age: false,
    income: false,
    savings: false,
    city: false,
    tenure: false,
    housingCost: false,
    householdSize: false,
  });

  // City autocomplete state (static client-side list → backend resolver → PUMAs).
  // The list is a static asset at /city_list.json; initCityList() warms its
  // module-level cache on mount, searchCities() reads from it synchronously.
  const [cityQuery, setCityQuery] = useState('');
  const [cityOptions, setCityOptions] = useState<CityOption[]>([]);
  const [selectedCity, setSelectedCity] = useState<ResolvedCity | null>(null);
  const [showCityDropdown, setShowCityDropdown] = useState(false);
  const [cityResolving, setCityResolving] = useState(false);
  const [cityError, setCityError] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Carousel state
  const [currentSlide, setCurrentSlide] = useState(0); // 0 = form, 1 = plan

  // Sign up modal state
  const [showSignUpModal, setShowSignUpModal] = useState(false);
  const [isModalClosing, setIsModalClosing] = useState(false);
  const [isModalOpening, setIsModalOpening] = useState(false);
  const [signUpData, setSignUpData] = useState({
    emailOrPhone: '',
    password: '',
    confirmPassword: ''
  });
  const [signUpErrors, setSignUpErrors] = useState({
    emailOrPhone: false,
    password: false,
    confirmPassword: false,
    passwordMatch: false
  });

  // Ref for smooth scrolling
  const formSectionRef = useRef<HTMLDivElement>(null);

  // Add login modal state
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [isLoginModalClosing, setIsLoginModalClosing] = useState(false);
  const [isLoginModalOpening, setIsLoginModalOpening] = useState(false);
  const [loginData, setLoginData] = useState({
    email: '',
    password: ''
  });
  const [loginErrors, setLoginErrors] = useState({
    email: false,
    password: false
  });

  // Add planning type state
  const [showPlanningQuestion, setShowPlanningQuestion] = useState(true);
  const [planningType, setPlanningType] = useState<'individual' | 'family'>('individual');

  // Load theme preference from localStorage on component mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      setIsDarkMode(savedTheme === 'dark');
    }
    if (localStorage.getItem('isAdmin') === 'true') {
      setIsAdmin(true);
    }
    setIsThemeLoaded(true);
  }, []);

  // Warm the city-list cache once on mount. Fires in the background — the
  // user has to click "Start Planning", pick Individual/Family, and fill
  // several fields before reaching the city input, so the fetch is almost
  // always finished by then. If not, searchCities returns [] until it is.
  useEffect(() => {
    initCityList();
  }, []);

  // Close dropdown on outside click.
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!dropdownRef.current?.contains(e.target as Node)) {
        setShowCityDropdown(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Synchronous client-side search — no debounce, no network during typing.
  function handleCityInput(e: React.ChangeEvent<HTMLInputElement>) {
    const val = e.target.value;
    setCityQuery(val);
    setSelectedCity(null);
    setCityError(false);
    if (val.length >= 2) {
      const results = searchCities(val);
      setCityOptions(results);
      setShowCityDropdown(results.length > 0);
    } else {
      setCityOptions([]);
      setShowCityDropdown(false);
    }
  }

  // Network happens only on city selection.
  async function handleCitySelect(opt: CityOption) {
    setShowCityDropdown(false);
    setCityQuery(opt.label);
    setCityResolving(true);
    try {
      const resolved = await resolveCity(opt);
      setSelectedCity(resolved);
      setCityError(false);
    } catch {
      setCityError(true);
      setSelectedCity(null);
    } finally {
      setCityResolving(false);
    }
  }

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const toggleDarkMode = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    // Save theme preference to localStorage
    localStorage.setItem('theme', newMode ? 'dark' : 'light');
  };

  const scrollToForm = () => {
    formSectionRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  };

  const validateForm = () => {
    const hhInt = parseInt(householdSize);
    const newErrors = {
      name: !name.trim(),
      age: !age || parseInt(age) <= 0 || parseInt(age) > 120,
      income: !income || parseFloat(income) < 0,
      savings: !savings || parseFloat(savings) < 0,
      city: selectedCity === null,
      tenure: tenure === null,
      housingCost: !housingCost || parseFloat(housingCost) <= 0,
      householdSize:
        planningType === 'family'
          ? !householdSize || Number.isNaN(hhInt) || hhInt < 2
          : false,
    };
    setErrors(newErrors);
    return !Object.values(newErrors).some(error => error);
  };

  // Screen 1 → Screen 2: gate on the existing personal-info validation;
  // advancing does NOT submit (submission happens from Screen 2).
  const handleNext = () => {
    if (!validateForm() || !selectedCity || !tenure) {
      const firstErrorField = document.querySelector(
        'input[style*="border-bottom: 2px solid rgb(239, 68, 68)"]'
      ) as HTMLElement | null;
      firstErrorField?.focus();
      return;
    }
    setSubmitError('');
    setCurrentScreen(2);
  };

  // The form's submit dispatcher. Routes by screen so Enter and the footer
  // button do the right thing: advance on Screen 1, submit on Screen 2.
  const handleFormSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (currentScreen === 1) {
      handleNext();
      return;
    }
    // Defensive: Screen 1 is already valid (Next gated it), but if something
    // changed, bounce back rather than submitting an invalid personal-info set.
    if (!validateForm() || !selectedCity || !tenure) {
      setCurrentScreen(1);
      return;
    }
    void handleSubmit(event);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!validateForm() || !selectedCity || !tenure) {
      // Scroll-to-first-error-field UX (matches the prior handleCreatePlan behavior).
      const firstErrorField = document.querySelector(
        'input[style*="border-bottom: 2px solid rgb(239, 68, 68)"]'
      ) as HTMLElement | null;
      firstErrorField?.focus();
      return;
    }

    const household_size =
      planningType === 'individual' ? 1 : parseInt(householdSize);

    // Debt: skip-all forces 0s (explicit "no debt"); otherwise blank → 0,
    // which the backend reads as "use cohort prior" (byte-identical to today).
    const debt = skipAllDebt
      ? { cc: 0, sl: 0, auto: 0, other: 0 }
      : {
          cc: parseFloat(ccCarriedBalance) || 0,
          sl: parseFloat(studentLoanPayment) || 0,
          auto: parseFloat(autoLoanPayment) || 0,
          other: parseFloat(otherDebtPayment) || 0,
        };

    const payload = {
      age: parseInt(age),
      gross_income: parseFloat(income),
      city_pumas: selectedCity.pumas,
      city_label: selectedCity.label,
      place_fips: selectedCity.place_fips,
      county_fips: selectedCity.county_fips,
      tenure,
      housing_cost: parseFloat(housingCost),
      household_size,
      savings: parseFloat(savings) || 0,
      filing_status: 'single',
      cc_carried_balance: debt.cc,
      student_loan_payment: debt.sl,
      auto_loan_payment: debt.auto,
      other_debt_payment: debt.other,
    };

    setSubmitError('');
    // Show the loading view and advance its staged copy on a timer. A minimum
    // display window keeps the messages readable even if the local backend
    // answers in a few hundred ms; the timer caps at the final phase (no loop).
    setIsLoading(true);
    setLoadingPhase(0);
    const startedAt = Date.now();
    const MIN_DISPLAY_MS = 2800;
    const PHASE_MS = 850;
    const phaseTimer = setInterval(() => {
      setLoadingPhase((p) => Math.min(p + 1, LOADING_PHASES.length - 1));
    }, PHASE_MS);
    try {
      const response = await fetch('http://localhost:8000/api/profiles/analyze/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const body = await response.text();
        setSubmitError(`Server error (${response.status}): ${body.slice(0, 200)}`);
        setIsLoading(false);
        return;
      }

      const data: ProfileAnalysis = await response.json();
      // Hold the loading view until the staged copy has had its minimum airtime,
      // then swap straight to results in one paint (no intermediate flash).
      const elapsed = Date.now() - startedAt;
      if (elapsed < MIN_DISPLAY_MS) {
        await new Promise((resolve) => setTimeout(resolve, MIN_DISPLAY_MS - elapsed));
      }
      setAnalysis(data);
      setShowResults(true);
      setIsLoading(false);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : 'Network error');
      setIsLoading(false);
    } finally {
      clearInterval(phaseTimer);
    }
  };

  // Get theme colors and styles
  const colors = getColors(isDarkMode);
  const buttonStyles = getButtonStyles(isDarkMode, colors);
  const toggleStyles = getToggleStyles(isDarkMode);

  const getInputBorderColor = (hasError: boolean) => {
    return hasError ? '#ef4444' : colors.accent;
  };

const validateSignUpForm = () => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
    const isValidEmailOrPhone = emailRegex.test(signUpData.emailOrPhone) || phoneRegex.test(signUpData.emailOrPhone);

    const newErrors = {
      emailOrPhone: !signUpData.emailOrPhone.trim() || !isValidEmailOrPhone,
      password: !signUpData.password || signUpData.password.length < 6,
      confirmPassword: !signUpData.confirmPassword,
      passwordMatch: signUpData.password !== signUpData.confirmPassword
    };

    setSignUpErrors(newErrors);
    return !Object.values(newErrors).some(error => error);
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateSignUpForm()) {
      return;
    }

    // Here you would typically make an API call to your backend
    console.log('Sign up data:', signUpData);
    alert('Sign up successful! (This would normally create an account)');
    setShowSignUpModal(false);

    // Reset form
    setSignUpData({
      emailOrPhone: '',
      password: '',
      confirmPassword: ''
    });
    setSignUpErrors({
      emailOrPhone: false,
      password: false,
      confirmPassword: false,
      passwordMatch: false
    });
  };

  const handleGoogleSignUp = () => {
    // Here you would integrate with Google OAuth
    alert('Google OAuth integration would go here!');
  };

  const getSignUpInputBorderColor = (hasError: boolean) => {
    return hasError ? '#ef4444' : '#d4a574';
  };

  // Function to open modal with animation
  const openSignUpModal = () => {
    setShowSignUpModal(true);
    setIsModalOpening(true);
    setTimeout(() => {
      setIsModalOpening(false);
    }, 150); // Very quick fade in
  };

  // Function to close modal with animation
  const closeSignUpModal = () => {
    setIsModalClosing(true);
    setTimeout(() => {
      setShowSignUpModal(false);
      setIsModalClosing(false);
    }, 150); // Very quick fade out
  };

  // Login modal functions
  const validateLoginForm = () => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const newErrors = {
      email: !loginData.email.trim() || !emailRegex.test(loginData.email),
      password: !loginData.password || loginData.password.length < 6
    };
    setLoginErrors(newErrors);
    return !Object.values(newErrors).some(error => error);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateLoginForm()) {
      return;
    }
    // Client-side admin gate (demo). Real auth isn't wired yet; this flips the
    // local `isAdmin` flag that unlocks the developer view on the results page.
    // Credentials: admin@finance.app / devadmin.
    const isAdminLogin =
      loginData.email.trim().toLowerCase() === 'admin@finance.app' &&
      loginData.password === 'devadmin';
    if (isAdminLogin) {
      setIsAdmin(true);
      localStorage.setItem('isAdmin', 'true');
      closeLoginModal();
      setLoginData({ email: '', password: '' });
      setLoginErrors({ email: false, password: false });
      return;
    }
    // Here you would typically make an API call to your backend
    console.log('Login data:', loginData);
    alert('Login successful! (This would normally authenticate the user)');
    setShowLoginModal(false);
    // Reset form
    setLoginData({ email: '', password: '' });
    setLoginErrors({ email: false, password: false });
  };

  const handleGoogleLogin = () => {
    // Here you would integrate with Google OAuth
    alert('Google OAuth login integration would go here!');
  };

  const getLoginInputBorderColor = (hasError: boolean) => {
    return hasError ? '#ef4444' : '#d4a574';
  };

  // Function to open login modal with animation
  const openLoginModal = () => {
    setShowLoginModal(true);
    setIsLoginModalOpening(true);
    setTimeout(() => {
      setIsLoginModalOpening(false);
    }, 150);
  };

  // Function to close login modal with animation
  const closeLoginModal = () => {
    setIsLoginModalClosing(true);
    setTimeout(() => {
      setShowLoginModal(false);
      setIsLoginModalClosing(false);
    }, 150);
  };

  const selectPlanningType = (type: 'individual' | 'family') => {
    setPlanningType(type);
    setShowPlanningQuestion(false);
  };

  // Don't render anything until theme is loaded to prevent flash
  if (!isThemeLoaded) {
    return null;
  }

  // Loading view — Swiss-minimal: precision spinner, numeric phase index,
  // cross-fading staged copy, and a determinate accent hairline. Gated before
  // results so the swap to the plan happens in a single paint.
  if (isLoading) {
    const total = LOADING_PHASES.length;
    const pct = ((loadingPhase + 1) / total) * 100;
    const trackColor = isDarkMode ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.08)';
    const mutedColor = isDarkMode ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)';
    return (
      <div
        style={{
          minHeight: '100vh',
          width: '100%',
          background: colors.background,
          color: colors.text,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 6vw',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '40px',
            width: '100%',
            maxWidth: '360px',
            textAlign: 'center',
          }}
        >
          {/* Thin precision spinner */}
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '50%',
              // longhands only (top = accent, rest = track) so the shorthand
              // `border` never conflicts with `borderTopColor` on re-render.
              borderWidth: '2px',
              borderStyle: 'solid',
              borderColor: `${colors.accent} ${trackColor} ${trackColor} ${trackColor}`,
              animation: 'spin 0.8s linear infinite',
            }}
          />

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '14px',
              width: '100%',
            }}
          >
            {/* Numeric phase index (Swiss numbering) */}
            <div
              style={{
                fontFamily: 'var(--font-geist-mono), monospace',
                fontSize: '11px',
                letterSpacing: '0.28em',
                textTransform: 'uppercase',
                color: colors.accent,
              }}
            >
              {String(loadingPhase + 1).padStart(2, '0')}&nbsp;&nbsp;/&nbsp;&nbsp;{String(total).padStart(2, '0')}
            </div>

            {/* Cycling status copy — keyed to re-trigger the fade on each change */}
            <div
              key={loadingPhase}
              className="plan-phase-text"
              style={{
                fontFamily: 'Georgia, "Times New Roman", serif',
                fontSize: '22px',
                fontWeight: 500,
                lineHeight: 1.3,
                color: colors.text,
                minHeight: '30px',
              }}
            >
              {LOADING_PHASES[loadingPhase]}
            </div>
          </div>

          {/* Determinate-feeling progress hairline */}
          <div
            style={{
              width: '100%',
              height: '1px',
              background: trackColor,
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                height: '100%',
                width: `${pct}%`,
                background: colors.accent,
                transition: 'width 0.7s cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            />
          </div>

          {/* Quiet reassurance line */}
          <div
            style={{
              fontFamily: 'var(--font-geist-sans), Arial, sans-serif',
              fontSize: '12px',
              letterSpacing: '0.04em',
              color: mutedColor,
            }}
          >
            Building your plan — this only takes a moment
          </div>
        </div>
      </div>
    );
  }

  if (showResults && analysis) {
    // Default result = the polished, user-facing money map. The raw dev view
    // below is reachable only for a signed-in admin who has toggled into it.
    if (!(isAdmin && devView)) {
      return (
        <ResultsView
          analysis={analysis}
          colors={colors}
          isDarkMode={isDarkMode}
          grossIncomeInput={income}
          name={name.trim() || undefined}
          onStartOver={() => { setShowResults(false); setAnalysis(null); setCurrentScreen(1); }}
          onToggleTheme={toggleDarkMode}
          isAdmin={isAdmin}
          onOpenDevView={() => setDevView(true)}
          onExitAdmin={() => {
            setIsAdmin(false);
            localStorage.removeItem('isAdmin');
            setDevView(false);
          }}
        />
      );
    }

    const fmt = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    // Unit toggle: monthly divides every dollar value by 12 (categories +
    // financial summary). BENEFITS are already $/mo and MATCH has no $, so
    // both are left as-is.
    const unit = devUnit === 'monthly' ? 12 : 1;
    const moneyCol = (n: number) => `$${fmt(n / unit).padStart(7)}`;   // value column
    const moneyInline = (n: number) => `$${fmt(n / unit)}`;            // bracket detail
    const isCollapsed = (g: string) => devCollapsed[g] ?? true;        // collapsed by default
    const toggleGroup = (g: string) =>
      setDevCollapsed((prev) => ({ ...prev, [g]: !(prev[g] ?? true) }));

    const dists = analysis.distributions;
    // Topic-grouped structure comes straight from the backend display_rollup
    // (single source of truth). Children sorted by predicted value desc; collapse
    // is display-only — the parent total always sums its children. Episodic
    // cadence (for italics) also comes from the roll-up, not a local set.
    //
    // **Build-4/6 totals** — the displayed value of a category is the back-fill
    // SPLIT: feasibility_adjusted (cohort-typical measured) + backfill_inferred
    // (the discretionary-accounting increment when the back-fill fires). Both
    // are real spending dollars; both must be summed everywhere here or the
    // four-way accounting (income − spend − slack = 0 on `primary`) breaks
    // by exactly Σ inferred. ``catValue`` is the canonical per-line getter.
    const catValue = (c: string) =>
      (dists[c]?.feasibility_adjusted ?? 0) + (dists[c]?.backfill_inferred ?? 0);
    const episodicSet = new Set<string>();
    const seen = new Set<string>();
    // Omit-by-default cats (chrty/educ/ocash/stdint/othint/finpay) are flagged
    // omit_from_initial_view by the backend (value-layer-zeroed). Track them so
    // they're excluded from BOTH the grouped render and the "ungrouped" drift
    // banner — they're intentionally absent, not lost. A group whose members are
    // all omitted (e.g. financial_debt) drops out via the children.length filter.
    const omittedCats = new Set<string>();
    // Defensive: if the API response predates the display_rollup wiring (e.g. the
    // Django backend wasn't restarted), fall back to no groups so every category
    // surfaces under "ungrouped" with a loud banner instead of crashing.
    const rollupTopics = analysis.display_rollup?.topics ?? [];
    const groups = rollupTopics
      .map((t) => {
        t.members.forEach((mm) => {
          if (mm.cadence === 'episodic') episodicSet.add(mm.category);
          if (mm.omit_from_initial_view) omittedCats.add(mm.category);
        });
        const children = t.members
          .filter((mm) => !mm.omit_from_initial_view)
          .map((mm) => mm.category)
          .filter((c) => c in dists)
          .sort((a, b) => catValue(b) - catValue(a));
        children.forEach((c) => seen.add(c));
        const total = children.reduce((s, c) => s + catValue(c), 0);
        return { group: t.topic, total, children, isNetWorth: !t.is_spending };
      })
      .filter((g) => g.children.length > 0);
    // Any distribution key the roll-up didn't place = backend partition drift —
    // surfaced loudly and NOT folded into the grouped total (reconciliation fails).
    // Omitted cats are excluded here (intentionally absent, not drift).
    const ungrouped = Object.keys(dists)
      .filter((c) => !seen.has(c) && !omittedCats.has(c))
      .sort((a, b) => catValue(b) - catValue(a));
    const groupedTotal = groups.reduce((s, g) => s + g.total, 0);
    const flatSum = Object.keys(dists).reduce((s, c) => s + catValue(c), 0);
    const reconcileOk = Math.abs(groupedTotal - flatSum) < 0.5;
    const nCats = Object.keys(dists).length;
    // Per-profile car-ownership prediction (match_metadata) — shown on the
    // transportation_travel group line. Ambiguous shows the probability (the
    // common urban-average case), per the car-owner band logic in algorithm.py.
    const cm = analysis.match_metadata;
    const carPred =
      cm.car_owner_classification === 'owner' ? 'car'
      : cm.car_owner_classification === 'non_owner' ? 'no car'
      : `~${Math.round(cm.car_owner_probability * 100)}% car`;

    // Per-category child line — the existing bracket detail, unit-aware.
    // cohort (a 0–1 ratio) and structural (bool) are not divided.
    //
    // Value column = measured (`feasibility_adjusted`) + inferred (`backfill_inferred`)
    // so reconciliation matches the group total. When inferred > 0, the bracket
    // surfaces the split `(meas + inf)` so the user can see where the
    // back-fill contribution came from.
    const childLine = (cat: string, d: SpendingDistribution) => {
      const inferred = d.backfill_inferred ?? 0;
      const value = d.feasibility_adjusted + inferred;
      const splitTag = inferred > 0
        ? ` ⟨meas:${moneyInline(d.feasibility_adjusted)} +inf:${moneyInline(inferred)}⟩`
        : '';
      return `    ${cat.padEnd(10)} ${moneyCol(value)}${splitTag}  ` +
        `[p10:${moneyInline(d.p10)} p25:${moneyInline(d.p25)} p50:${moneyInline(d.p50)} ` +
        `p75:${moneyInline(d.p75)} p90:${moneyInline(d.p90)} ` +
        `| engel:${moneyInline(d.engel_estimate)} cohort:${d.cohort_position.toFixed(2)} ` +
        `structural:${d.is_structural} gap:${moneyInline(d.behavioral_gap)}]`;
    };

    // Build-5/6 four-way accounting (paycheck-deductions + spend + savings +
    // remainder). Surfaced here so the dev view shows the full picture and
    // the user can reconcile every dollar of take-home to a likely
    // destination (COMPLETE-DOLLAR-ACCOUNTING).
    //
    // The "Committed outflows" bucket label is jargon — expanded inline into
    // per-item rows (only non-zero items shown) so each line is self-
    // explanatory. The bucket holds things that come off-the-top before the
    // user has real discretion: retirement contribution, health insurance
    // premium employee share, HSA / FSA / commuter pre-tax set-asides,
    // supplemental life+disability. Debt service is a sibling line, not
    // strictly payroll-deducted but contractually required.
    const committedItems = (analysis.committed_outflows?.items ?? [])
      .filter((it) => it.annual > 0);
    const committed = analysis.committed_outflows?.total_annual ?? 0;
    const debtSvc = analysis.debt_service_annual ?? 0;
    const savings = analysis.residual_assignment?.savings_investment.annual ?? 0;
    const remainder = analysis.residual_assignment?.genuine_remainder.annual ?? 0;
    // State-dependent savings copy (predict-not-prescribe): acknowledge the
    // user's reported balance where it informed the prediction; never suggest
    // they save more. `signal_would_pull_up_deferred` shows the cohort number
    // with a neutral "your balance suggests higher — adjust if accurate"
    // affordance (acknowledging-data, not prescribing-action).
    const savingsFraming =
      analysis.residual_assignment?.savings_investment.framing_state ??
      'signal_confirmed_cohort';
    const savingsLabel = ({
      signal_confirmed_cohort: 'Est. savings',
      signal_pulled_down: 'Est. saving (from your balance — adjust if savings held elsewhere)',
      signal_would_pull_up_deferred: 'Est. saving (balance suggests higher — adjust if accurate)',
      signal_pulled_up_routed: 'Est. saving (routed to tax-advantaged accounts)',
      user_pinned: 'Your saving (you adjusted this)',
    } as Record<string, string>)[savingsFraming] ?? 'Est. savings';
    // === Take-home redefinition (the bank-account number) ===
    // "Take-home" = gross − taxes − PRE-TAX contributions (post-tax AND
    // post-pre-tax-contribution). The pre-tax committed outflows (those that
    // reduced AGI) move OUT of the four-way and INTO the tax wedge — otherwise a
    // user sees their 401(k) subtracted from a "take-home" that should already be
    // net of it. DISPLAY-ONLY: d_variable + the allocation are UNTOUCHED.
    const tb = analysis.tax_breakdown;
    const grossAnnual = Number.isFinite(parseFloat(income))
      ? parseFloat(income)
      : (tb ? tb.take_home + tb.total_tax : 0);
    // Pre-tax contributions = gross − AGI = the income-tax-excludable wedge the
    // fixed point computed (committed pre-tax baseline + the waterfall 401k/HSA
    // top-ups). This is the exact set that moves into the wedge.
    const pretaxContrib = tb ? Math.max(0, grossAnnual - tb.federal_agi) : 0;
    const preTaxItems = committedItems.filter((it) => it.pre_tax);
    const postTaxItems = committedItems.filter((it) => !it.pre_tax);
    // Pre-tax committed at DISPLAY value (incl the waterfall top-ups surfaced as
    // predicted_topup) — the wedge lines; they sum to gross−AGI (partition check).
    const preTaxWedgeTotal = preTaxItems.reduce((s, it) => s + (it.display_annual ?? it.annual), 0);
    const postTaxCommitted = postTaxItems.reduce((s, it) => s + (it.display_annual ?? it.annual), 0);
    // The waterfall's PRE-TAX fills (401k/HSA top-ups = Σ committed predicted_topup)
    // also belong in the wedge; only the POST-tax fills (IRA/taxable/cc_paydown) stay
    // in the four-way waterfall term.
    const itTopup = committedItems.reduce((s, it) => s + (it.predicted_topup_annual ?? 0), 0);
    const wf = analysis.savings_waterfall;
    const waterfall = wf?.total ?? 0;
    const postTaxWaterfall = waterfall - itTopup;
    // NEW take-home = surfaced (gross−taxes) minus the pre-tax contributions.
    const backOutTakeHome = analysis.d_variable_annual + committed;
    const takeHomeAnnual = tb ? (tb.take_home - pretaxContrib) : backOutTakeHome;
    // Cross-check that PROVES d_variable didn't move: the new take-home must equal
    // d_variable + committed − pre-tax-contributions (all on the UNCHANGED
    // d_variable + committed). Divergence = a computation drifted (a tripwire).
    const takeHomeCrossOk = tb == null
      || Math.abs(takeHomeAnnual - (analysis.d_variable_annual + committed - pretaxContrib)) < 1;
    // Four-way from the NEW take-home: ONLY post-tax committed + debt + spending +
    // savings + post-tax waterfall + remainder. Spending/savings/debt/remainder
    // dollars are byte-identical; only the take-home line and the placement of the
    // pre-tax outflows changed.
    const fourWaySum = postTaxCommitted + debtSvc + flatSum + savings + postTaxWaterfall + remainder;
    const fourWayOk = Math.abs(takeHomeAnnual - fourWaySum) < 1 &&
                     (analysis.structural_deficit ?? 0) < 1;

    // Per-item paycheck-deduction lines. Strip the parenthetical
    // "(cohort-typical, adjustable)" suffix and the redundant
    // "— employee share" suffix the backend includes on some labels (dev view
    // knows it's the employee share without being told).
    const cleanLabel = (s: string) => s
      .replace(/\s*\([^)]*\)\s*$/, '')
      .replace(/\s*[—–]\s*employee share\s*$/i, '')
      .trim();
    // Four-way committed lines = POST-tax committed only (supplemental life/
    // disability). The pre-tax committed (401k/§125) moved into the tax wedge.
    const paycheckLines = postTaxItems.map((it) =>
      `  − ${cleanLabel(it.label).padEnd(40)} ${moneyCol(it.display_annual ?? it.annual)}`
    );

    // Back-fill indicator block — when fired, show which categories were
    // lifted and by how much. Reads `analysis.backfill.inferred` (a dict of
    // category → inferred annual $; only nonzero entries are present). Sorted
    // by amount desc. Human-readable labels for the back-fill target codes.
    const bf = analysis.backfill;
    const backfillLabels: Record<string, string> = {
      entertainment: 'Entertainment',
      shopping: 'Shopping',
      hotel: 'Hotels & motels',
      airshp: 'Air & ship travel',
      recrp: 'Recreation products',
    };
    const backfillEntries = bf?.fired
      ? Object.entries(bf.inferred).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])
      : [];
    const backfillTotal = backfillEntries.reduce((s, [, v]) => s + v, 0);
    // Per-category before/after detail for the spending-categories section.
    // `before` = the measured cohort-typical `feasibility_adjusted` the allocator
    // produced; `+backfill` = the `backfill_inferred` increment the post-allocation
    // discretionary-accounting stage lifted it by; `after` = what the category line
    // above actually shows. Sums to `backfillTotal`.
    const backfillBeforeTotal = backfillEntries.reduce(
      (s, [c]) => s + (dists[c]?.feasibility_adjusted ?? 0), 0);
    const backfillDetailLines = backfillEntries.length > 0
      ? [
          `BACK-FILL DETAIL (${devUnit} $) — which categories the discretionary`,
          `accounting stage lifted, and by how much (before → +backfill → after):`,
          `  ${'category'.padEnd(28)} ${'before'.padStart(8)} ${'backfill'.padStart(8)} ${'after'.padStart(8)}`,
          ...backfillEntries.map(([cat, inf]) => {
            const before = dists[cat]?.feasibility_adjusted ?? 0;
            return `  ${(backfillLabels[cat] ?? cat).padEnd(28)} ` +
              `${moneyCol(before)} ${moneyCol(inf)} ${moneyCol(before + inf)}`;
          }),
          `  ${'TOTAL'.padEnd(28)} ${moneyCol(backfillBeforeTotal)} ` +
            `${moneyCol(backfillTotal)} ${moneyCol(backfillBeforeTotal + backfillTotal)}`,
          `  (pool ${moneyInline(bf?.pool ?? 0)}; s* ${((bf?.s_star ?? 0) * 100).toFixed(1)}%` +
            // Spend-arm: when a low reported balance pulled the back-fill
            // benchmark below cohort s*, show the personalized rate used.
            ((bf?.s_star_personalized ?? bf?.s_star ?? 0) < (bf?.s_star ?? 0) - 1e-9
              ? ` → ${((bf?.s_star_personalized ?? 0) * 100).toFixed(1)}% balance-informed`
              : '') +
            // Stratification: name which cap population the lifts read.
            (bf?.stratified
              ? `; hi-earner caps (cp90_hi, threshold y_eq ${moneyInline(bf?.cohort_median_y_eq ?? 0)})`
              : '') +
            ` — cap-bound below pool means more capacity would push past cohort p90)`,
        ]
      : [`BACK-FILL DETAIL: not fired (solver_status=${analysis.solver_status}; ` +
         `no discretionary categories lifted for this profile).`];
    const backfillLines = backfillEntries.length > 0
      ? [
          '',
          'BACK-FILL (discretionary lifted to cohort-realistic):',
          ...backfillEntries.map(([cat, v]) =>
            `  + ${(backfillLabels[cat] ?? cat).padEnd(40)} ${moneyCol(v)}`
          ),
          `  ${''.padEnd(40)} ${moneyCol(backfillTotal)} total lifted` +
            ` (pool ${moneyInline(bf?.pool ?? 0)}; cap-bound below pool means more discretionary capacity would push past cohort p90)`,
        ]
      : [];

    // Savings waterfall section (7c3, 2026-06-10) — fires on high-contradiction
    // (full 401k→IRA→HSA→taxable walk) and no-contradiction (taxable-only default).
    // Rendered as its own section so the fills credit the four-way sum without
    // touching committed lines (un-topped `annual` stays the committed subtraction;
    // double-counting the 401k/HSA topups in both sections is the wrong direction).
    const waterfallFillLabels: Record<string, string> = {
      cc_paydown: 'CC paydown (accelerated)',
      k401_topup: '401(k) top-up',
      ira: 'IRA (backdoor Roth)',
      hsa_topup: 'HSA top-up',
      taxable_savings: 'Taxable savings',
    };
    // The PRE-tax fills (401k/HSA top-ups) are shown in the tax wedge (they're
    // pre-tax contributions); the four-way waterfall section shows only the
    // POST-tax fills (IRA / taxable / CC paydown), summing to postTaxWaterfall.
    const postTaxFills = (wf?.fills ?? []).filter(
      (f) => f.code !== 'k401_topup' && f.code !== 'hsa_topup'
    );
    const waterfallLines = wf?.fired && postTaxFills.length > 0
      ? [
          '',
          `SAVINGS WATERFALL (${wf.trigger}) — ${moneyInline(postTaxWaterfall)} routed post-tax ` +
            `(limits ${wf.limits_year}; 401k/HSA top-ups shown in the tax wedge):`,
          ...postTaxFills.map((f) =>
            `  + ${(waterfallFillLabels[f.code] ?? f.label).padEnd(40)} ${moneyCol(f.annual)}` +
            (f.maxed ? '  [maxed]' : '') +
            (f.mechanism ? `  (${f.mechanism})` : '')
          ),
        ]
      : [];

    // Debt-accumulation annotation (soft-deficit-with-CC-debt) — conditional
    // and OUT of the four-way sum: the compressed allocation above already
    // sums to the post-debt budget, so this line never enters RECONCILE.
    // Copy per framing_state (Q4 wording, predict-not-prescribe: states the
    // math + conditional consequence + correction pathway; never "you should",
    // never "tight spot"). The dollar is the FULL gap — the hedge lives in the
    // words. When the solver compressed but the signal didn't fire (no carried
    // CC balance), surface that as a one-line diagnostic (dev view only).
    const da = analysis.debt_accumulation;
    const daGap = `${moneyInline(da?.annual_potential_growth ?? 0)}${devUnit === 'monthly' ? '/mo' : '/yr'}`;
    const daCopy: Record<string, string[]> = {
      signal_clear: [
        `  Spending typical for your income and area would run about ${daGap} above`,
        `  your take-home after debt payments. If your actual spending is close to`,
        `  typical, that gap would likely add to your credit card balance. If you`,
        `  spend less than typical, or pay more toward your cards than the minimum,`,
        `  adjust the figures so they match your situation.`,
      ],
      signal_marginal: [
        `  Typical spending for your income would run a little above your take-home`,
        `  after debt payments — about ${daGap}. Whether that becomes added balance`,
        `  depends on your actual spending. Adjust if your numbers differ.`,
      ],
      user_pinned: [
        `  Shown as you set it (your adjustment replaced the cohort-typical figure).`,
      ],
    };
    const debtAccumLines = da?.applies
      ? [
          '',
          `DEBT ACCUMULATION (conditional — outside the four-way sum) [${da.framing_state}]:`,
          `  Potential added CC balance${' '.padEnd(14)} ${moneyCol(da.annual_potential_growth ?? 0)}`,
          ...(daCopy[da.framing_state ?? 'signal_marginal'] ?? []),
          `  (gap = ${((da.gap_ratio ?? 0) * 100).toFixed(1)}% of post-debt budget; ` +
            `CC balance = ${((da.cc_balance_to_income ?? 0) * 100).toFixed(1)}% of income)`,
        ]
      : analysis.solver_status === 'soft_constrained'
        ? [
            '',
            'DEBT ACCUMULATION: not fired (compressed profile, but no carried CC ' +
              'balance reported above the trivial floor).',
          ]
        : [];

    // Per-debt-type breakdown (Hybrid B framing — surface debt prominently).
    // Initial view shows ONLY user-reported debt (omit_from_initial_view === false);
    // cohort-predicted + not-modeled lines are omitted (the omit-by-default
    // treatment). ``userDebt`` feeds both the four-way summary lines below and the
    // "financial_debt" section in the spending breakdown (user choice: debt renders
    // under financial-debt). Falls back to the aggregate line when the response has
    // no per-type liabilities (pre-debt-build API).
    const liabilities = analysis.balance_sheet?.liabilities ?? {};
    const sourceTag: Record<string, string> = {
      user_reported: 'from your input',
      cohort_predicted: 'est. typical',
      not_modeled: '',
    };
    const userDebt = ['othdbt', 'stddbt', 'auto_loan', 'other_debt']
      .map((k) => liabilities[k])
      .filter((l): l is NonNullable<typeof l> =>
        !!l && !l.omit_from_initial_view && l.annual_service > 0);
    const userDebtTotal = userDebt.reduce((s, l) => s + l.annual_service, 0);
    const debtComponentLines = userDebt.map((l) => {
      const tag = sourceTag[l.source] ? `  [${sourceTag[l.source]}]` : '';
      return `  − ${l.label.padEnd(27)} ${moneyCol(l.annual_service)}${tag}`;
    });
    const debtLines = debtComponentLines.length > 0
      ? debtComponentLines
      : [`  − Debt service${' '.padEnd(25)} ${moneyCol(debtSvc)}`];

    // Gross→take-home tax wedge. Take-home is the BANK-ACCOUNT number: gross minus
    // taxes AND the pre-tax contributions (the ones that reduced AGI). Three
    // tripwires (each a redundant-computation guard, same discipline as the
    // four-way RECONCILE): (1) RECONCILE — the displayed deductions (taxes + pre-tax
    // contributions) must sum to gross − take-home; (2) partition — the pre-tax
    // lines must sum to gross − AGI (the fixed point's taxable-income reduction);
    // (3) cross-check — take-home == d_variable + committed − pre-tax (proves
    // d_variable did NOT move; this is a display-only change). grossAnnual is the
    // user's income input (the same value the backend used).
    const fu = analysis.filing_unit;
    const taxLinesTotal = tb
      ? tb.federal_tax + tb.fica + tb.state_tax + tb.state_payroll_tax + tb.city_tax
      : 0;
    const wedgeDeductions = taxLinesTotal + preTaxWedgeTotal;  // taxes + pre-tax contributions
    const wedgeReconcileOk = tb != null && Math.abs(wedgeDeductions - (grossAnnual - takeHomeAnnual)) < 1;
    const partitionOk = tb != null && Math.abs(preTaxWedgeTotal - pretaxContrib) < 1;
    const taxWedgeLines = tb ? [
      `Gross income:        ${moneyCol(grossAnnual)}`,
      `  − Federal income tax${' '.padEnd(11)} ${moneyCol(tb.federal_tax)}` +
        (tb.federal_tax < 0 ? '  (net refund — refundable EITC/CTC)' : ''),
      `  − FICA (SS + Medicare)${' '.padEnd(9)} ${moneyCol(tb.fica)}`,
      `  − State income tax${' '.padEnd(13)} ${moneyCol(tb.state_tax)}` +
        (tb.state_eitc > 0 ? `  (net of state EITC ${moneyInline(tb.state_eitc)})` : ''),
      ...(tb.state_payroll_tax > 0
        ? [`  − State payroll (SDI/PFML)${' '.padEnd(5)} ${moneyCol(tb.state_payroll_tax)}`] : []),
      ...(tb.city_tax > 0
        ? [`  − Local / municipal${' '.padEnd(12)} ${moneyCol(tb.city_tax)}`] : []),
      // Pre-tax contributions (reduced AGI) — moved OUT of the four-way INTO here,
      // shown at their display value (incl. any waterfall top-up). Σ == gross − AGI.
      ...preTaxItems.map((it) =>
        `  − ${cleanLabel(it.label).padEnd(38)} ${moneyCol(it.display_annual ?? it.annual)}` +
        ((it.predicted_topup_annual ?? 0) > 0 ? '  (incl. top-up; pre-tax)' : '  (pre-tax)')),
      `  = Take-home (bank):  ${moneyCol(takeHomeAnnual)}`,
      `RECONCILE: Σ deductions ${moneyInline(wedgeDeductions)} ${wedgeReconcileOk ? '==' : '!='} ` +
        `gross − take-home ${moneyInline(grossAnnual - takeHomeAnnual)}` +
        (wedgeReconcileOk ? '  ✓'
          : `  ✗ Δ${moneyInline(wedgeDeductions - (grossAnnual - takeHomeAnnual))} — a displayed line diverged`),
      `partition:  Σ pre-tax ${moneyInline(preTaxWedgeTotal)} ${partitionOk ? '==' : '!='} ` +
        `gross − AGI ${moneyInline(pretaxContrib)}` +
        (partitionOk ? '  ✓' : `  ✗ Δ${moneyInline(preTaxWedgeTotal - pretaxContrib)}`),
      `cross-check: take-home ${moneyInline(takeHomeAnnual)} ${takeHomeCrossOk ? '==' : '!='} ` +
        `d_var + committed − pre-tax ${moneyInline(analysis.d_variable_annual + committed - pretaxContrib)}` +
        `${takeHomeCrossOk ? '  ✓ (d_variable unchanged)' : '  ✗ DRIFT'}`,
      ``,
      `federal: AGI ${moneyInline(tb.federal_agi)}  ` +
        `${tb.itemized ? `itemized ${moneyInline(tb.itemized_deductions)}` : `standard ${moneyInline(tb.standard_deduction)}`}` +
        `${tb.federal_amt > 0 ? `  AMT ${moneyInline(tb.federal_amt)}` : ''}` +
        `${tb.federal_eitc > 0 ? `  EITC ${moneyInline(tb.federal_eitc)}` : ''}` +
        `${tb.federal_ctc > 0 ? `  CTC ${moneyInline(tb.federal_ctc)}` : ''}` +
        `  eff rate ${(tb.effective_rate * 100).toFixed(1)}%`,
      ...(fu ? [
        ``,
        `assumed filing unit (the calculator's one imputed input — adjust household`,
        `size / filing status on the form to correct; the tax self-corrects):`,
        `  filing status: ${fu.filing_status}${fu.imputed ? '  (estimated from your inputs)' : '  (as entered)'}`,
        `  dependents: ${fu.num_dependents}` +
          `${fu.dependent_ages.length ? `  (ages est. ~${fu.dependent_ages.join(', ')})` : ''}` +
          `${fu.eic_qualifying_children ? `  EIC-qualifying: ${fu.eic_qualifying_children}` : ''}`,
      ] : []),
    ] : [];

    const summaryLines = [
      `Zone:                ${analysis.financial_zone}`,
      `Solver status:       ${analysis.solver_status}`,
      `Take-home (bank):    ${moneyCol(takeHomeAnnual)}  (${devUnit}; net of taxes + pre-tax contributions — see tax wedge)`,
      ...paycheckLines,
      ...debtLines,
      `  − Spending (incl. back-fill)${' '.padEnd(12)} ${moneyCol(flatSum)}`,
      `  − ${savingsLabel.padEnd(38)} ${moneyCol(savings)}`,
      ...waterfallLines,
      `  = Genuine remainder${' '.padEnd(20)} ${moneyCol(remainder)}`,
      `RECONCILE (four-way): ${moneyInline(fourWaySum)} ${fourWayOk ? '==' : '!='} take_home ${moneyInline(takeHomeAnnual)}` +
        (fourWayOk ? '  ✓' : `  ✗ Δ${moneyInline(takeHomeAnnual - fourWaySum)}` +
          ((analysis.structural_deficit ?? 0) > 0
            ? ` (structural_deficit ${moneyInline(analysis.structural_deficit)} explains overage — compressed profile)`
            : '')),
      ...backfillLines,
      ...debtAccumLines,
      '',
      `D_variable:          ${moneyCol(analysis.d_variable_annual)}  (post-deductions, what the allocator sees)`,
      `Pace / slack:        ${moneyCol(analysis.feasibility_slack)}  (= est. savings + genuine remainder)`,
      `Structural deficit:  ${moneyCol(analysis.structural_deficit)}`,
    ].join('\n');

    // BENEFITS + MATCH — unchanged (benefits already $/mo; match has no $).
    const benefitMatchLines = [
      `BENEFITS (${analysis.benefits.length}):`,
      ...(analysis.benefits.length === 0
        ? ['(none)']
        : analysis.benefits.map(b =>
            `${b.program_name}  $${fmt(b.estimated_monthly_min)}-$${fmt(b.estimated_monthly_max)}/mo  ` +
            `confidence:${b.confidence}  ${b.enrollment_url}`
          )),
      ``,
      `MATCH:`,
      `n_effective:    ${analysis.match_metadata.n_effective.toFixed(1)}`,
      `n_households:   ${analysis.match_metadata.n_households}`,
      `confidence:     ${analysis.match_metadata.confidence}`,
      `city_pumas:     ${analysis.match_metadata.city_pumas_used.join(', ')}`,
      `pumas_used:     ${analysis.match_metadata.pumas_used.join(', ')}`,
    ].join('\n');

    const rowStyle = { whiteSpace: 'pre-wrap', wordBreak: 'break-word' } as const;
    const tabStyle = (active: boolean) => ({
      fontFamily: 'inherit',
      fontSize: '0.8rem',
      padding: '0.3rem 0.9rem',
      cursor: 'pointer',
      border: `1px solid ${colors.text}`,
      backgroundColor: active ? colors.text : 'transparent',
      color: active ? colors.background : colors.text,
      fontWeight: active ? 700 : 400,
    });

    return (
      <div style={{ backgroundColor: colors.background, color: colors.text, minHeight: '100vh', padding: '2vh 2vw', fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace', fontSize: '0.8rem' }}>
        <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', marginBottom: '1.5vh' }}>
            <h2 style={{ fontFamily: 'inherit', margin: 0, fontSize: '1rem' }}>Profile analysis (raw — dev view)</h2>
            <button
              onClick={() => setDevView(false)}
              style={{ ...tabStyle(false), borderRadius: '4px' }}
            >
              ← Back to overview
            </button>
          </div>

          {/* Unit toggle */}
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5vh' }}>
            {(['monthly', 'annual'] as const).map((u) => (
              <button key={u} onClick={() => setDevUnit(u)} style={tabStyle(devUnit === u)}>
                {u}
              </button>
            ))}
          </div>

          {/* Gross→take-home tax wedge (Stage 6) — collapsible, ahead of the
              Take-home line, mirroring the toggleGroup idiom. The RECONCILE +
              cross-check inside are the frontend's tripwires (same discipline as
              the four-way RECONCILE); a ✗ surfaces even collapsed. */}
          {tb && (
            <div style={{ marginBottom: '1vh' }}>
              <div
                onClick={() => toggleGroup('tax_wedge')}
                style={{ ...rowStyle, cursor: 'pointer' }}
              >
                {`${isCollapsed('tax_wedge') ? '▸' : '▾'} GROSS → TAKE-HOME (tax wedge)` +
                  `   ${moneyCol(grossAnnual)} → ${moneyCol(takeHomeAnnual)}` +
                  `   eff ${(tb.effective_rate * 100).toFixed(1)}%` +
                  `${(!wedgeReconcileOk || !takeHomeCrossOk || !partitionOk) ? '   ✗ RECONCILE FAILED' : ''}`}
              </div>
              {!isCollapsed('tax_wedge') && (
                <pre style={{ ...rowStyle, margin: 0 }}>{taxWedgeLines.join('\n')}</pre>
              )}
            </div>
          )}

          <pre style={{ ...rowStyle, margin: 0 }}>{summaryLines}</pre>

          {/* Topic-grouped spending — groups collapsed by default; click to expand */}
          <div style={{ marginTop: '1.5vh' }}>
            <div style={rowStyle}>{`SPENDING CATEGORIES (${devUnit} $, ${nCats} total, grouped):`}</div>
            {groups.map((g) => (
              <div key={g.group}>
                <div
                  onClick={() => toggleGroup(g.group)}
                  style={{ ...rowStyle, cursor: 'pointer' }}
                >
                  {`${isCollapsed(g.group) ? '▸' : '▾'} ${g.group.padEnd(22)} ${moneyCol(g.total)}` +
                    `${g.isNetWorth ? '  (non-spending / balance)' : ''}  (${g.children.length})` +
                    `${g.group === 'transportation_travel' ? `  — [predicted: ${carPred}]` : ''}`}
                </div>
                {!isCollapsed(g.group) && g.children.map((cat) => (
                  <div
                    key={cat}
                    style={{ ...rowStyle, fontStyle: episodicSet.has(cat) ? 'italic' : 'normal' }}
                  >
                    {childLine(cat, dists[cat])}
                  </div>
                ))}
              </div>
            ))}
            {/* financial_debt — the cohort spending members here (educ/chrty/etc.)
                are omit-by-default, so the group is empty; surface the user's
                REPORTED debt-service under this heading instead (user choice).
                These are debt-service $/yr, NOT spending — excluded from the
                spending RECONCILE total below (which sums only the groups). */}
            {userDebt.length > 0 && (
              <div>
                <div
                  onClick={() => toggleGroup('financial_debt')}
                  style={{ ...rowStyle, cursor: 'pointer' }}
                >
                  {`${isCollapsed('financial_debt') ? '▸' : '▾'} ${'financial_debt'.padEnd(22)} ${moneyCol(userDebtTotal)}` +
                    `  (${userDebt.length})  — your reported debt service (not in spending total)`}
                </div>
                {!isCollapsed('financial_debt') && userDebt.map((l) => (
                  <div key={l.label} style={rowStyle}>
                    {`  − ${l.label.padEnd(27)} ${moneyCol(l.annual_service)}` +
                      `  [${sourceTag[l.source] ?? l.source}]`}
                  </div>
                ))}
              </div>
            )}
            {ungrouped.length > 0 && (
              <div>
                <div style={{ ...rowStyle, fontWeight: 700 }}>
                  {rollupTopics.length === 0
                    ? `ungrouped (${ungrouped.length}) — API response has no display_rollup; RESTART the Django backend`
                    : `ungrouped (PARTITION/JOIN BUG — ${ungrouped.length} not placed by display_rollup):`}
                </div>
                {ungrouped.map((cat) => (
                  <div key={cat} style={rowStyle}>{childLine(cat, dists[cat])}</div>
                ))}
              </div>
            )}
            <div style={{ ...rowStyle, marginTop: '0.5rem' }}>
              {`RECONCILE: Σ group totals ${moneyInline(groupedTotal)} ${reconcileOk ? '==' : '!='} ` +
                `Σ all ${nCats} categories ${moneyInline(flatSum)}` +
                (reconcileOk ? '  ✓ lossless' : `  ✗ MISMATCH Δ${moneyInline(groupedTotal - flatSum)} — partition/join bug`)}
            </div>
            <pre style={{ ...rowStyle, marginTop: '1rem' }}>{backfillDetailLines.join('\n')}</pre>
          </div>

          <pre style={{ ...rowStyle, marginTop: '1.5vh' }}>{benefitMatchLines}</pre>

          <button
            onClick={() => { setShowResults(false); setAnalysis(null); setCurrentScreen(1); }}
            style={{ ...buttonStyles.primary, minWidth: '160px', marginTop: '3vh' }}
          >
            Start over
          </button>
        </div>
      </div>
    );
  }

  // ---- Shared form inputs (used in both Individual and Family sentences) ----

  const cityInput = (
    <div
      ref={dropdownRef}
      style={{ position: 'relative', display: 'inline-block' }}
    >
      <input
        type="text"
        value={cityQuery}
        onChange={handleCityInput}
        className={(errors.city || cityError) ? 'error' : ''}
        style={{
          width: '220px',
          padding: '0.125vh 0.5vw 0.25vh 0.5vw',
          borderWidth: '0 0 2px',
          borderStyle: 'solid',
          borderColor: cityError
            ? '#ef4444'
            : cityResolving
            ? colors.text
            : selectedCity
            ? colors.accent
            : errors.city
            ? '#ef4444'
            : colors.accent,
          backgroundColor: 'transparent',
          color: colors.accent,
          fontWeight: '600',
          outline: 'none',
          textAlign: 'center',
          fontSize: '1.25rem',
          fontFamily: 'Georgia, "Times New Roman", serif',
          opacity: cityResolving ? 0.6 : 1,
          transition: 'opacity 0.2s ease, border-color 0.2s ease',
        }}
        placeholder="Chicago, IL"
      />
      {showCityDropdown && cityOptions.length > 0 && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            minWidth: '240px',
            backgroundColor: colors.background,
            border: `1px solid ${colors.accent}`,
            borderRadius: '0.5rem',
            zIndex: 100,
            maxHeight: '240px',
            overflowY: 'auto',
          }}
        >
          {cityOptions.map((opt, i) => (
            <div
              key={i}
              onClick={() => handleCitySelect(opt)}
              style={{
                paddingTop: '12px',
                paddingBottom: '12px',
                paddingLeft: '28px',
                paddingRight: '16px',
                textIndent: '-12px',
                minHeight: '44px',
                cursor: 'pointer',
                color: colors.text,
                fontSize: '1rem',
                lineHeight: '1.4',
                textAlign: 'left',
                fontFamily: 'Georgia, "Times New Roman", serif',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = colors.accent; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = colors.text; }}
            >
              {opt.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderTenureLine = (pronoun: 'I' | 'We') => (
    <div style={{ marginBottom: '2vh', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5vw', flexWrap: 'wrap' }}>
      <span style={{ color: colors.text }}>{pronoun}</span>
      <span
        onClick={() => setTenure('RENT')}
        style={{
          color: tenure === 'RENT' ? colors.accent : (errors.tenure ? '#ef4444' : colors.accent),
          textDecoration: 'underline',
          cursor: 'pointer',
          fontWeight: tenure === 'RENT' ? '700' : '500',
          transition: 'color 0.3s ease',
        }}
      >
        rent
      </span>
      <span style={{ color: colors.text }}>/</span>
      <span
        onClick={() => setTenure('OWN')}
        style={{
          color: tenure === 'OWN' ? colors.accent : (errors.tenure ? '#ef4444' : colors.accent),
          textDecoration: 'underline',
          cursor: 'pointer',
          fontWeight: tenure === 'OWN' ? '700' : '500',
          transition: 'color 0.3s ease',
        }}
      >
        own
      </span>
      <span style={{ color: colors.text }}>and pay</span>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <span
          style={{
            position: 'absolute',
            left: '8px',
            top: '50%',
            transform: 'translateY(-50%)',
            color: colors.text,
            fontSize: '1.25rem',
            fontFamily: 'Georgia, "Times New Roman", serif',
            fontWeight: '600',
            pointerEvents: 'none',
            zIndex: 1,
          }}
        >$</span>
        <input
          type="number"
          value={housingCost}
          onChange={(e) => setHousingCost(e.target.value)}
          className={errors.housingCost ? 'error' : ''}
          style={{
            width: '120px',
            padding: '0.125vh 0.5vw 0.25vh 20px',
            borderWidth: '0 0 2px',
            borderStyle: 'solid',
            borderColor: getInputBorderColor(errors.housingCost),
            backgroundColor: 'transparent',
            color: colors.accent,
            fontWeight: '600',
            outline: 'none',
            textAlign: 'center',
            fontSize: '1.25rem',
            fontFamily: 'Georgia, "Times New Roman", serif',
          }}
          placeholder="1500"
        />
      </div>
      <span style={{ color: colors.text }}>per month.</span>
    </div>
  );

  // Universal Dark/Light Mode Toggle Button that appears on all pages
  const DarkModeToggle = (
    <div style={toggleStyles.container}>
      <button
        onClick={toggleDarkMode}
        style={toggleStyles.button}
        title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      >
        <div style={toggleStyles.circle} />
      </button>
    </div>
  );

  return (
    <div style={{
      fontFamily: 'var(--font-geist-sans)',
      backgroundColor: colors.background,
      color: colors.text,
      transition: 'background-color 0.3s ease, color 0.3s ease',
      scrollSnapType: 'y mandatory',
      height: '100vh',
      overflowY: 'scroll',
      scrollBehavior: 'smooth'
    }}>
      {/* Header - Only on first section */}
      <header style={{
        width: '100%',
        padding: '10px 0',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        backgroundColor: 'transparent',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '50px' }}>
          {/* Hamburger Menu Button */}
          <button
            onClick={toggleMenu}
            style={{
              background: 'transparent',
              border: 'none',
              color: isMenuOpen ? colors.accent : colors.text,
              cursor: 'pointer',
              padding: '10px',
              marginLeft: '20px',
              marginTop: '0',
              transition: 'color 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '44px',
              width: '44px'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = colors.accent;
              e.currentTarget.style.cursor = 'pointer';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = isMenuOpen ? colors.accent : colors.text;
              e.currentTarget.style.cursor = 'pointer';
            }}
          >
            <svg style={{ width: '24px', height: '24px', cursor: 'pointer' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          {/* Log In button */}
          <div style={{
            display: 'flex',
            gap: '0.8vw',
            marginRight: '20px',
            marginTop: '0',
            alignItems: 'center',
            position: 'relative',
            height: '50px'
          }}>
            {/* Login Button */}
            <button
              onClick={openLoginModal}
              style={{
                ...buttonStyles.secondary,
                height: '40px',
                fontSize: '14px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '0 16px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = colors.accent;
                e.currentTarget.style.borderColor = colors.accent;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = colors.text;
                e.currentTarget.style.borderColor = colors.border;
              }}
            >
              Log In
            </button>
          </div>
        </div>
      </header>

      {/* Horizontal Menu */}
      {isMenuOpen && (
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '70px',
          zIndex: 20,
          animation: 'slideInFromLeft 0.3s ease-out',
          height: '44px',
          display: 'flex',
          alignItems: 'center',
          backgroundColor: colors.background,
          borderRadius: '0.6vh',
          padding: '0 2vw'
        }}>
          <nav style={{ display: 'flex', gap: '3vw', alignItems: 'center', height: '100%' }}>
            <button
              onClick={() => {
                // Scroll to top of page (home section)
                window.scrollTo({ top: 0, behavior: 'smooth' });
                toggleMenu();
              }}
              style={{
                fontSize: '16px',
                fontWeight: '500',
                color: colors.text,
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                transition: 'color 0.3s ease',
                padding: '0 16px',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                paddingTop: '2px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = colors.accent;
                e.currentTarget.style.cursor = 'pointer';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = colors.text;
                e.currentTarget.style.cursor = 'pointer';
              }}
            >
              Home
            </button>
            <button
              onClick={() => {
                router.push('/about');
                toggleMenu();
              }}
              style={{
                fontSize: '16px',
                fontWeight: '500',
                color: colors.text,
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                transition: 'color 0.3s ease',
                padding: '0 16px',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                paddingTop: '2px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = colors.accent;
                e.currentTarget.style.cursor = 'pointer';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = colors.text;
                e.currentTarget.style.cursor = 'pointer';
              }}
            >
              About
            </button>
          </nav>
        </div>
      )}

      {/* Overlay for mobile menu */}
      {isMenuOpen && (
        <div
          style={{
            position: 'absolute',
            top: '7vh',
            left: 0,
            right: 0,
            bottom: 0
          }}
          onClick={toggleMenu}
        ></div>
      )}

      {/* Section 1: Home Content */}
      <section style={{
        ...layoutStyles.fullSection,
        backgroundColor: colors.background
      }}>
        <div style={layoutStyles.contentContainer}>
          <main style={layoutStyles.flexColumn}>
            <Image
              style={{ filter: isDarkMode ? 'invert(1)' : 'none', marginTop: '-2vh' }}
              src="/placeholder.png"
              alt="Placeholder logo"
              width={200}
              height={100}
              priority
            />
            <RotatingText isDarkMode={isDarkMode} />
            <p style={{ color: colors.text, marginTop: '-1.6vh' }}>Project your short and long term goals in two minutes.</p>
            <button
              onClick={scrollToForm}
              style={buttonStyles.primary}
              onMouseEnter={(e) => {
                if (isDarkMode) {
                  e.currentTarget.style.backgroundColor = colors.accent;
                  e.currentTarget.style.borderColor = colors.accent;
                } else {
                  e.currentTarget.style.color = colors.accent;
                  e.currentTarget.style.borderColor = colors.accent;
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
              onMouseLeave={(e) => {
                if (isDarkMode) {
                  e.currentTarget.style.backgroundColor = colors.buttonBg;
                  e.currentTarget.style.borderColor = colors.buttonBg;
                } else {
                  e.currentTarget.style.color = colors.text;
                  e.currentTarget.style.borderColor = colors.text;
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              Start Planning
            </button>
          </main>
        </div>
      </section>

      {/* Section 2: Form/Plan Container */}
      <section
        ref={formSectionRef}
        style={{
          minHeight: '100vh',
          backgroundColor: colors.background,
          color: colors.text,
          scrollSnapAlign: 'start',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'background-color 0.3s ease, color 0.3s ease',
          position: 'relative'
        }}
      >
        <style dangerouslySetInnerHTML={{
          __html: `
            input[type="number"]::-webkit-outer-spin-button,
            input[type="number"]::-webkit-inner-spin-button {
              -webkit-appearance: none;
              margin: 0;
            }
            input[type="number"] {
              -moz-appearance: textfield;
            }
            input::placeholder {
              color: #d4a574;
              opacity: 0.38;
            }
            input::-webkit-input-placeholder {
              color: #d4a574;
              opacity: 0.38;
            }
            input::-moz-placeholder {
              color: #d4a574;
              opacity: 0.38;
            }
            input:-ms-input-placeholder {
              color: #d4a574;
              opacity: 0.38;
            }
            
            /* Error state placeholder styles */
            input.error::placeholder {
              color: #ef4444;
              opacity: 0.6;
            }
            input.error::-webkit-input-placeholder {
              color: #ef4444;
              opacity: 0.6;
            }
            input.error::-moz-placeholder {
              color: #ef4444;
              opacity: 0.6;
            }
            input.error:-ms-input-placeholder {
              color: #ef4444;
              opacity: 0.6;
            }
          `
        }} />

        {/* Form View - Always visible initially */}
        <div style={{
          width: '100%',
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'absolute',
          top: 0,
          left: 0,
          transform: currentSlide === 1 ? 'translateX(-100%)' : 'translateX(0%)',
          transition: 'transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
          zIndex: 1
        }}>
          <div style={{ maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
            <main style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '4.8vh',
              alignItems: 'center',
              justifyContent: 'center',
              width: '100%',
              padding: '0 1.6vw'
            }}>
              <h1 style={{
                fontSize: '2.5rem',
                color: colors.text,
                fontFamily: 'Georgia, "Times New Roman", serif',
                fontWeight: '400',
                letterSpacing: '-0.02em',
                textAlign: 'center',
                margin: 0,
                marginBottom: '0'
              }}>Let&apos;s get to know you better!</h1>

              {/* Container that matches the height of first section's content before button */}
              <div style={{
                minHeight: '24.4vh',
                width: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'flex-start',
                position: 'relative',
                padding: '0 1.6vw'
              }}>
                {/* Individual | Family toggle — shown only on Screen 1 (the
                    personal-info step). Hidden on Screen 2 (debt), where the
                    planning type is already chosen and the form is identical
                    for both. Clicking either reveals the respective form. */}
                {currentScreen === 1 && (
                <div style={{
                  display: 'flex',
                  marginTop: '10px',
                  marginBottom: '30px',
                  justifyContent: 'center',
                  width: '100%'
                }}>
                  <Segmented
                    value={showPlanningQuestion ? null : planningType}
                    options={[{ v: 'individual', label: 'Individual' }, { v: 'family', label: 'Family' }]}
                    onChange={selectPlanningType}
                    width={240}
                    height={40}
                    fontSize={14}
                    accent={colors.accent}
                    text={isDarkMode ? '#ffffff' : '#666666'}
                    hairline={isDarkMode ? 'rgba(255, 255, 255, 0.25)' : 'rgba(0, 0, 0, 0.14)'}
                  />
                </div>
                )}

                {!showPlanningQuestion && (
                  <div style={{ animation: 'fadeIn 0.25s ease-out', width: '100%' }}>
                    <form id="financial-form" onSubmit={handleFormSubmit} style={{ width: '100%' }}>
                      {currentScreen === 1 && (
                      <div style={{
                        fontSize: '1.25rem',
                        lineHeight: '1.6',
                        textAlign: 'center',
                        fontFamily: 'Georgia, "Times New Roman", serif'
                      }}>
                        {planningType === 'individual' ? (
                          // Individual form
                          <>
                            <div style={{ marginBottom: '2vh', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1vw' }}>
                              <span style={{ color: colors.text }}>Hi! My name is</span>
                              <input
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                className={errors.name ? 'error' : ''}
                                style={{
                                  width: '180px',
                                  padding: '0.125vh 0.5vw 0.25vh 0.5vw',
                                  borderWidth: '0 0 2px',
                                  borderStyle: 'solid',
                                  borderColor: getInputBorderColor(errors.name),
                                  backgroundColor: 'transparent',
                                  color: colors.accent,
                                  fontWeight: '600',
                                  outline: 'none',
                                  textAlign: 'center',
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif'
                                }}
                                placeholder="your name"
                              />
                              <span style={{ color: colors.text }}>and I am</span>
                              <input
                                type="number"
                                value={age}
                                onChange={(e) => setAge(e.target.value)}
                                className={errors.age ? 'error' : ''}
                                style={{
                                  width: '80px',
                                  padding: '0.125vh 0.5vw 0.25vh 0.5vw',
                                  borderWidth: '0 0 2px',
                                  borderStyle: 'solid',
                                  borderColor: getInputBorderColor(errors.age),
                                  backgroundColor: 'transparent',
                                  color: colors.accent,
                                  fontWeight: '600',
                                  outline: 'none',
                                  textAlign: 'center',
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif'
                                }}
                                placeholder="25"
                              />
                              <span style={{ color: colors.text }}>years old.</span>
                            </div>

                            <div style={{ marginBottom: '2vh', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5vw' }}>
                              <span style={{ color: colors.text }}>I currently make</span>
                              <div style={{ position: 'relative', display: 'inline-block' }}>
                                <span style={{
                                  position: 'absolute',
                                  left: '8px',
                                  top: '50%',
                                  transform: 'translateY(-50%)',
                                  color: colors.text,
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif',
                                  fontWeight: '600',
                                  pointerEvents: 'none',
                                  zIndex: 1
                                }}>$</span>
                                <input
                                  type="number"
                                  value={income}
                                  onChange={(e) => setIncome(e.target.value)}
                                  className={errors.income ? 'error' : ''}
                                  style={{
                                    width: '140px',
                                    padding: '0.125vh 0.5vw 0.25vh 20px',
                                    borderWidth: '0 0 2px',
                                    borderStyle: 'solid',
                                    borderColor: getInputBorderColor(errors.income),
                                    backgroundColor: 'transparent',
                                    color: colors.accent,
                                    fontWeight: '600',
                                    outline: 'none',
                                    textAlign: 'center',
                                    fontSize: '1.25rem',
                                    fontFamily: 'Georgia, "Times New Roman", serif'
                                  }}
                                  placeholder="50000"
                                />
                              </div>
                              <span style={{ color: colors.text }}>per year and have</span>
                              <div style={{ position: 'relative', display: 'inline-block' }}>
                                <span style={{
                                  position: 'absolute',
                                  left: '8px',
                                  top: '50%',
                                  transform: 'translateY(-50%)',
                                  color: colors.text,
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif',
                                  fontWeight: '600',
                                  pointerEvents: 'none',
                                  zIndex: 1
                                }}>$</span>
                                <input
                                  type="number"
                                  value={savings}
                                  onChange={(e) => setSavings(e.target.value)}
                                  className={errors.savings ? 'error' : ''}
                                  style={{
                                    width: '140px',
                                    padding: '0.125vh 0.5vw 0.25vh 20px',
                                    borderWidth: '0 0 2px',
                                    borderStyle: 'solid',
                                    borderColor: getInputBorderColor(errors.savings),
                                    backgroundColor: 'transparent',
                                    color: colors.accent,
                                    fontWeight: '600',
                                    outline: 'none',
                                    textAlign: 'center',
                                    fontSize: '1.25rem',
                                    fontFamily: 'Georgia, "Times New Roman", serif'
                                  }}
                                  placeholder="10000"
                                />
                              </div>
                              <span style={{ color: colors.text }}>in savings.</span>
                            </div>

                            {renderTenureLine('I')}

                            <div style={{ marginBottom: '2vh', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1vw' }}>
                              <span style={{ color: colors.text }}>I live in</span>
                              {cityInput}
                              <span style={{ color: colors.text }}>and I&apos;m ready to start planning my financial future!</span>
                            </div>
                          </>
                        ) : (
                          // Family form
                          <>
                            <div style={{ marginBottom: '2vh', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1vw' }}>
                              <span style={{ color: colors.text }}>Hi! My name is</span>
                              <input
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                className={errors.name ? 'error' : ''}
                                style={{
                                  width: '180px',
                                  padding: '0.125vh 0.5vw 0.25vh 0.5vw',
                                  borderWidth: '0 0 2px',
                                  borderStyle: 'solid',
                                  borderColor: getInputBorderColor(errors.name),
                                  backgroundColor: 'transparent',
                                  color: colors.accent,
                                  fontWeight: '600',
                                  outline: 'none',
                                  textAlign: 'center',
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif'
                                }}
                                placeholder="your name"
                              />
                              <span style={{ color: colors.text }}>and I am</span>
                              <input
                                type="number"
                                value={age}
                                onChange={(e) => setAge(e.target.value)}
                                className={errors.age ? 'error' : ''}
                                style={{
                                  width: '80px',
                                  padding: '0.125vh 0.5vw 0.25vh 0.5vw',
                                  borderWidth: '0 0 2px',
                                  borderStyle: 'solid',
                                  borderColor: getInputBorderColor(errors.age),
                                  backgroundColor: 'transparent',
                                  color: colors.accent,
                                  fontWeight: '600',
                                  outline: 'none',
                                  textAlign: 'center',
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif'
                                }}
                                placeholder="35"
                              />
                              <span style={{ color: colors.text }}>years old.</span>
                            </div>

                            <div style={{ marginBottom: '2vh', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1vw' }}>
                              <span style={{ color: colors.text }}>We are a family of</span>
                              <input
                                type="number"
                                value={householdSize}
                                onChange={(e) => setHouseholdSize(e.target.value)}
                                className={errors.householdSize ? 'error' : ''}
                                style={{
                                  width: '80px',
                                  padding: '0.125vh 0.5vw 0.25vh 0.5vw',
                                  borderWidth: '0 0 2px',
                                  borderStyle: 'solid',
                                  borderColor: getInputBorderColor(errors.householdSize),
                                  backgroundColor: 'transparent',
                                  color: colors.accent,
                                  fontWeight: '600',
                                  outline: 'none',
                                  textAlign: 'center',
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif'
                                }}
                                placeholder="4"
                              />
                              <span style={{ color: colors.text }}>.</span>
                            </div>

                            <div style={{ marginBottom: '2vh', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5vw' }}>
                              <span style={{ color: colors.text }}>We currently make</span>
                              <div style={{ position: 'relative', display: 'inline-block' }}>
                                <span style={{
                                  position: 'absolute',
                                  left: '8px',
                                  top: '50%',
                                  transform: 'translateY(-50%)',
                                  color: colors.text,
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif',
                                  fontWeight: '600',
                                  pointerEvents: 'none',
                                  zIndex: 1
                                }}>$</span>
                                <input
                                  type="number"
                                  value={income}
                                  onChange={(e) => setIncome(e.target.value)}
                                  className={errors.income ? 'error' : ''}
                                  style={{
                                    width: '140px',
                                    padding: '0.125vh 0.5vw 0.25vh 20px',
                                    borderWidth: '0 0 2px',
                                    borderStyle: 'solid',
                                    borderColor: getInputBorderColor(errors.income),
                                    backgroundColor: 'transparent',
                                    color: colors.accent,
                                    fontWeight: '600',
                                    outline: 'none',
                                    textAlign: 'center',
                                    fontSize: '1.25rem',
                                    fontFamily: 'Georgia, "Times New Roman", serif'
                                  }}
                                  placeholder="120000"
                                />
                              </div>
                              <span style={{ color: colors.text }}>per year and have</span>
                              <div style={{ position: 'relative', display: 'inline-block' }}>
                                <span style={{
                                  position: 'absolute',
                                  left: '8px',
                                  top: '50%',
                                  transform: 'translateY(-50%)',
                                  color: colors.text,
                                  fontSize: '1.25rem',
                                  fontFamily: 'Georgia, "Times New Roman", serif',
                                  fontWeight: '600',
                                  pointerEvents: 'none',
                                  zIndex: 1
                                }}>$</span>
                                <input
                                  type="number"
                                  value={savings}
                                  onChange={(e) => setSavings(e.target.value)}
                                  className={errors.savings ? 'error' : ''}
                                  style={{
                                    width: '140px',
                                    padding: '0.125vh 0.5vw 0.25vh 20px',
                                    borderWidth: '0 0 2px',
                                    borderStyle: 'solid',
                                    borderColor: getInputBorderColor(errors.savings),
                                    backgroundColor: 'transparent',
                                    color: colors.accent,
                                    fontWeight: '600',
                                    outline: 'none',
                                    textAlign: 'center',
                                    fontSize: '1.25rem',
                                    fontFamily: 'Georgia, "Times New Roman", serif'
                                  }}
                                  placeholder="30000"
                                />
                              </div>
                              <span style={{ color: colors.text }}>in savings.</span>
                            </div>

                            {renderTenureLine('We')}

                            <div style={{ marginBottom: '2vh', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1vw' }}>
                              <span style={{ color: colors.text }}>We live in</span>
                              {cityInput}
                              <span style={{ color: colors.text }}>and we&apos;re ready to start planning our family&apos;s financial future!</span>
                            </div>
                          </>
                        )}
                      </div>
                      )}

                      {currentScreen === 2 && (() => {
                        // Structured aligned rows (Swiss): a two-line label on the
                        // left, the dollar input right-aligned in a shared column,
                        // hairline dividers between. Replaces the prior madlib voice.
                        const hairline = isDarkMode ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.10)';
                        const mutedColor = isDarkMode ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.45)';
                        const rowStyle = {
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: '1vw',
                          padding: '1.9vh 0',
                          borderBottom: `1px solid ${hairline}`,
                        };
                        const labelGroupStyle = {
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '3px',
                          textAlign: 'left',
                        } as const;
                        const mainLabelStyle = {
                          fontFamily: 'Georgia, "Times New Roman", serif',
                          fontSize: '1rem',
                          fontWeight: 500,
                          lineHeight: 1.2,
                          color: skipAllDebt ? mutedColor : colors.text,
                          transition: 'color 0.3s ease',
                        };
                        const subLabelStyle = {
                          fontFamily: 'var(--font-geist-sans), Arial, sans-serif',
                          fontSize: '0.78rem',
                          letterSpacing: '0.02em',
                          color: mutedColor,
                        };
                        // Three monthly-payment rows share an identical shape.
                        const loanRows = [
                          { key: 'sl', label: 'Student loans', value: studentLoanPayment, onChange: setStudentLoanPayment },
                          { key: 'auto', label: 'Car payments', value: autoLoanPayment, onChange: setAutoLoanPayment },
                          { key: 'other', label: 'Other debts', value: otherDebtPayment, onChange: setOtherDebtPayment },
                        ];
                        return (
                          <div style={{ maxWidth: '460px', margin: '0 auto' }}>
                            {/* Heading */}
                            <div style={{
                              marginBottom: '2.5vh',
                              textAlign: 'center',
                              fontFamily: 'Georgia, "Times New Roman", serif',
                              fontSize: '1.25rem',
                              color: colors.text,
                            }}>
                              Now, about any debts{' '}
                              <span style={{ opacity: 0.6, fontSize: '0.95rem', fontFamily: 'var(--font-geist-sans), Arial, sans-serif' }}>— skip what doesn&apos;t apply</span>
                            </div>

                            <div style={{ borderTop: `1px solid ${hairline}` }}>
                              {/* Credit-card carried balance — special: carries the tooltip. */}
                              <div style={rowStyle}>
                                <div style={labelGroupStyle}>
                                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.4vw' }}>
                                    <span style={mainLabelStyle}>Credit card balance</span>
                                    {/* Discoverable "?" tooltip — clarifies "carried" vs total.
                                        Hover gated to MOUSE pointers so a touch tap doesn't fire a
                                        synthetic enter that opens-then-the-click-toggle-closes it;
                                        on touch the button's onClick toggle owns open/close. */}
                                    <span
                                      style={{ position: 'relative', display: 'inline-flex' }}
                                      onPointerEnter={(e) => { if (e.pointerType === 'mouse') setCcTooltipOpen(true); }}
                                      onPointerLeave={(e) => { if (e.pointerType === 'mouse') setCcTooltipOpen(false); }}
                                    >
                                      <button
                                        type="button"
                                        aria-label="What is a carried balance?"
                                        onClick={() => setCcTooltipOpen((o) => !o)}
                                        style={{
                                          width: '18px', height: '18px', borderRadius: '50%',
                                          border: `1.5px solid ${colors.accent}`, color: colors.accent,
                                          backgroundColor: 'transparent', cursor: 'pointer',
                                          fontSize: '0.72rem', fontWeight: 700, lineHeight: 1,
                                          fontFamily: 'Georgia, serif', padding: 0,
                                          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                                        }}
                                      >?</button>
                                      {ccTooltipOpen && (
                                        <span
                                          role="tooltip"
                                          style={{
                                            position: 'absolute', top: '150%', left: '50%',
                                            transform: 'translateX(-50%)', width: '250px', zIndex: 10,
                                            backgroundColor: colors.text, color: colors.background,
                                            padding: '0.75rem 0.9rem', borderRadius: '8px',
                                            fontSize: '0.8rem', lineHeight: 1.45, fontWeight: 400,
                                            fontFamily: 'var(--font-geist-sans), Arial, sans-serif', textAlign: 'left',
                                            boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
                                          }}
                                        >
                                          The portion of your balance that doesn&apos;t get paid off — what carries
                                          over and accrues interest. Pay in full each month? Enter $0.
                                        </span>
                                      )}
                                    </span>
                                  </span>
                                  <span style={subLabelStyle}>carried month-to-month</span>
                                </div>
                                <DollarInput value={ccCarriedBalance} onChange={setCcCarriedBalance} placeholder="0" width={120} disabled={skipAllDebt} accent={colors.accent} text={colors.text} />
                              </div>

                              {/* Three monthly-payment rows. */}
                              {loanRows.map((r) => (
                                <div key={r.key} style={rowStyle}>
                                  <div style={labelGroupStyle}>
                                    <span style={mainLabelStyle}>{r.label}</span>
                                    <span style={subLabelStyle}>per month</span>
                                  </div>
                                  <DollarInput value={r.value} onChange={r.onChange} placeholder="0" width={120} disabled={skipAllDebt} accent={colors.accent} text={colors.text} />
                                </div>
                              ))}
                            </div>

                            {/* Skip-all fast-path. */}
                            <label style={{
                              display: 'flex', justifyContent: 'center', alignItems: 'center',
                              gap: '0.6vw', cursor: 'pointer', fontSize: '0.95rem',
                              color: colors.text, marginTop: '2.6vh',
                              fontFamily: 'var(--font-geist-sans), Arial, sans-serif',
                            }}>
                              <input
                                type="checkbox"
                                checked={skipAllDebt}
                                onChange={(e) => setSkipAllDebt(e.target.checked)}
                                style={{ width: '18px', height: '18px', cursor: 'pointer', accentColor: colors.accent }}
                              />
                              I don&apos;t have any debt — skip this
                            </label>
                          </div>
                        );
                      })()}
                    </form>
                  </div>
                )}
              </div>

              <div style={{ textAlign: 'center' }}>
                {showPlanningQuestion ? (
                  null
                ) : (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1vw' }}>
                      {currentScreen === 2 && (
                        <button
                          type="button"
                          onClick={() => { setSubmitError(''); setCurrentScreen(1); }}
                          style={{
                            ...buttonStyles.primary,
                            minWidth: '120px',
                            width: '120px',
                            backgroundColor: 'transparent',
                            color: colors.text,
                            borderColor: colors.text,
                          }}
                        >
                          ← Back
                        </button>
                      )}
                      <button
                        type="submit"
                        form="financial-form"
                        style={{
                          ...buttonStyles.primary,
                          minWidth: '180px',
                          width: '180px'
                        }}
                        onMouseEnter={(e) => {
                          if (isDarkMode) {
                            e.currentTarget.style.backgroundColor = colors.accent;
                            e.currentTarget.style.borderColor = colors.accent;
                          } else {
                            e.currentTarget.style.color = colors.accent;
                            e.currentTarget.style.borderColor = colors.accent;
                            e.currentTarget.style.backgroundColor = 'transparent';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (isDarkMode) {
                            e.currentTarget.style.backgroundColor = colors.buttonBg;
                            e.currentTarget.style.borderColor = colors.buttonBg;
                          } else {
                            e.currentTarget.style.color = colors.text;
                            e.currentTarget.style.borderColor = colors.text;
                            e.currentTarget.style.backgroundColor = 'transparent';
                          }
                        }}
                      >
                        {currentScreen === 1
                          ? 'Next →'
                          : `Create ${planningType === 'individual' ? 'My' : 'Our'} Plan`}
                      </button>
                    </div>
                    {submitError && (
                      <p style={{ color: '#ef4444', marginTop: '1.5vh', fontSize: '0.875rem' }}>
                        {submitError}
                      </p>
                    )}
                  </>
                )}
              </div>
            </main>
          </div>
        </div>

        {/* Plan View - Slides in from right */}
        <div style={{
          width: '100%',
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'absolute',
          top: 0,
          left: 0,
          transform: currentSlide === 1 ? 'translateX(0%)' : 'translateX(100%)',
          transition: 'transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
          padding: '4vh 2vw',
          zIndex: 2
        }}>
          <div style={{ maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
            {/* Header */}
            <div style={{ textAlign: 'center', marginBottom: '6vh' }}>
              <h1 style={{
                fontSize: '3rem',
                marginBottom: '2vh',
                color: colors.text,
                fontFamily: 'Georgia, "Times New Roman", serif',
                fontWeight: '400',
                letterSpacing: '-0.02em'
              }}>
                Your Financial Plan
              </h1>
              <p style={{
                fontSize: '1.25rem',
                color: colors.text,
                opacity: 0.8,
                fontFamily: 'Georgia, "Times New Roman", serif'
              }}>
                Here&apos;s a preview of your personalized financial insights
              </p>
            </div>

            {/* Placeholder Graphs Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '3vw',
              marginBottom: '6vh'
            }}>
              {/* Savings Growth Chart */}
              <div style={{
                padding: '3vh 2.5vw',
                border: `2px solid ${colors.accent}`,
                borderRadius: '1rem',
                backgroundColor: isDarkMode ? 'rgba(212, 165, 116, 0.08)' : 'rgba(212, 165, 116, 0.05)',
                textAlign: 'center'
              }}>
                <h3 style={{
                  color: colors.accent,
                  marginBottom: '2vh',
                  fontSize: '1.5rem',
                  fontFamily: 'Georgia, "Times New Roman", serif'
                }}>
                  Savings Growth Projection
                </h3>
                <div style={{
                  height: '200px',
                  backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
                  borderRadius: '0.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '2vh',
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  {/* Placeholder chart visualization */}
                  <div style={{
                    position: 'absolute',
                    bottom: '20px',
                    left: '20px',
                    right: '20px',
                    height: '60%',
                    background: `linear-gradient(45deg, ${colors.accent}40, ${colors.accent}80)`,
                    borderRadius: '4px',
                    clipPath: 'polygon(0% 100%, 20% 80%, 40% 85%, 60% 60%, 80% 40%, 100% 20%, 100% 100%)'
                  }} />
                  <span style={{
                    color: colors.text,
                    opacity: 0.6,
                    fontSize: '0.9rem',
                    zIndex: 1
                  }}>
                    Interactive Chart Preview
                  </span>
                </div>
                <p style={{
                  color: colors.text,
                  fontSize: '1rem',
                  opacity: 0.8
                }}>
                  Track your savings growth over time with personalized projections
                </p>
              </div>

              {/* Monthly Budget Breakdown */}
              <div style={{
                padding: '3vh 2.5vw',
                border: `2px solid ${colors.accent}`,
                borderRadius: '1rem',
                backgroundColor: isDarkMode ? 'rgba(212, 165, 116, 0.08)' : 'rgba(212, 165, 116, 0.05)',
                textAlign: 'center'
              }}>
                <h3 style={{
                  color: colors.accent,
                  marginBottom: '2vh',
                  fontSize: '1.5rem',
                  fontFamily: 'Georgia, "Times New Roman", serif'
                }}>
                  Monthly Budget
                </h3>
                <div style={{
                  height: '200px',
                  backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
                  borderRadius: '0.5rem',
                  display: 'flex',
                  alignItems: 'flex-end',
                  justifyContent: 'space-around',
                  padding: '20px',
                  marginBottom: '2vh'
                }}>
                  {/* Placeholder bar chart */}
                  {[60, 80, 45, 90, 35].map((height, index) => (
                    <div key={index} style={{
                      width: '20px',
                      height: `${height}%`,
                      backgroundColor: colors.accent,
                      borderRadius: '2px',
                      opacity: 0.7 + (index * 0.1)
                    }} />
                  ))}
                </div>
                <p style={{
                  color: colors.text,
                  fontSize: '1rem',
                  opacity: 0.8
                }}>
                  Optimized budget allocation for your financial goals
                </p>
              </div>

              {/* Goal Timeline */}
              <div style={{
                padding: '3vh 2.5vw',
                border: `2px solid ${colors.accent}`,
                borderRadius: '1rem',
                backgroundColor: isDarkMode ? 'rgba(212, 165, 116, 0.08)' : 'rgba(212, 165, 116, 0.05)',
                textAlign: 'center'
              }}>
                <h3 style={{
                  color: colors.accent,
                  marginBottom: '2vh',
                  fontSize: '1.5rem',
                  fontFamily: 'Georgia, "Times New Roman", serif'
                }}>
                  Goal Timeline
                </h3>
                <div style={{
                  height: '200px',
                  backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
                  borderRadius: '0.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '2vh',
                  position: 'relative'
                }}>
                  {/* Placeholder timeline */}
                  <div style={{
                    width: '80%',
                    height: '4px',
                    backgroundColor: colors.accent,
                    borderRadius: '2px',
                    position: 'relative'
                  }}>
                    {[25, 50, 75].map((position, index) => (
                      <div key={index} style={{
                        position: 'absolute',
                        left: `${position}%`,
                        top: '-6px',
                        width: '16px',
                        height: '16px',
                        backgroundColor: colors.accent,
                        borderRadius: '50%',
                        border: `3px solid ${colors.background}`
                      }} />
                    ))}
                  </div>
                </div>
                <p style={{
                  color: colors.text,
                  fontSize: '1rem',
                  opacity: 0.8
                }}>
                  Milestone tracking for your short and long-term objectives
                </p>
              </div>
            </div>

            {/* Sign Up Button and Back Button */}
            <div style={{ textAlign: 'center', marginTop: '6vh' }}>
              <p style={{
                fontSize: '1.25rem',
                color: colors.text,
                marginBottom: '3vh',
                fontFamily: 'Georgia, "Times New Roman", serif'
              }}>
                Ready to unlock your full financial potential?
              </p>
              <div style={{
                position: 'relative',
                width: '100%',
                display: 'flex',
                justifyContent: 'center'
              }}>
                <button
                  onClick={() => setCurrentSlide(0)}
                  style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    border: `1px solid ${colors.border}`,
                    backgroundColor: 'transparent',
                    color: colors.text,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    position: 'absolute',
                    left: 'calc(50% - 90px)',
                    transform: 'translateX(-100%)',
                    marginLeft: '20px'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = colors.accent;
                    e.currentTarget.style.borderColor = colors.accent;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = colors.text;
                    e.currentTarget.style.borderColor = colors.border;
                  }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="15 18 9 12 15 6"></polyline>
                  </svg>
                </button>
                <button
                  onClick={openSignUpModal}
                  style={{
                    ...buttonStyles.primary,
                    width: '120px',
                    height: '40px'
                  }}
                  onMouseEnter={(e) => {
                    if (isDarkMode) {
                      e.currentTarget.style.backgroundColor = colors.accent;
                      e.currentTarget.style.borderColor = colors.accent;
                    } else {
                      e.currentTarget.style.color = colors.accent;
                      e.currentTarget.style.borderColor = colors.accent;
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (isDarkMode) {
                      e.currentTarget.style.backgroundColor = colors.buttonBg;
                      e.currentTarget.style.borderColor = colors.buttonBg;
                    } else {
                      e.currentTarget.style.color = colors.text;
                      e.currentTarget.style.borderColor = colors.text;
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  Sign Up
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes slideInFromLeft {
            from {
              opacity: 0;
              transform: translateX(-2vw);
            }
            to {
              opacity: 1;
              transform: translateX(0);
            }
          }

          @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
          }
          
          @keyframes slideInFromRight {
            from {
              transform: translateX(100%);
            }
            to {
              transform: translateX(0%);
            }
          }
          
          /* Smooth scrolling with snap behavior */
          html {
            scroll-behavior: smooth;
          }
          
          /* Control scroll speed */
          * {
            scroll-behavior: smooth;
          }
          
          /* Smooth scroll container */
          body, html {
            scroll-snap-type: y mandatory;
            scroll-behavior: smooth;
          }
          
          /* Hide scrollbar for all browsers */
          ::-webkit-scrollbar {
            display: none;
            width: 0;
            height: 0;
          }
          
          /* Hide scrollbar for Firefox */
          html {
            scrollbar-width: none;
          }
          
          /* Hide scrollbar for IE and Edge */
          body {
            -ms-overflow-style: none;
          }
          
          /* Additional scrollbar hiding for main container */
          div {
            scrollbar-width: none;
            -ms-overflow-style: none;
          }
          
          div::-webkit-scrollbar {
            display: none;
            width: 0;
            height: 0;
          }
        `
      }} />

      {/* Sign Up Modal */}
      {showSignUpModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '2vh 2vw',
            opacity: isModalClosing ? 0 : (isModalOpening ? 0 : 1),
            transition: 'opacity 0.15s ease-out'
          }}
          onClick={closeSignUpModal}
        >
          <div
            style={{
              backgroundColor: colors.background,
              borderRadius: '1rem',
              padding: '4vh 3vw',
              maxWidth: '500px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              position: 'relative',
              border: `2px solid ${colors.accent}`,
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
              transform: isModalClosing ? 'scale(0.95)' : (isModalOpening ? 'scale(0.95)' : 'scale(1)'),
              transition: 'transform 0.15s ease-out'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={closeSignUpModal}
              style={{
                position: 'absolute',
                top: '1vh',
                right: '0.5vw',
                background: 'transparent',
                border: 'none',
                color: colors.text,
                fontSize: '1.2rem',
                cursor: 'pointer',
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                transition: 'background-color 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              ×
            </button>

            {/* Modal Header */}
            <div style={{ textAlign: 'center', marginBottom: '4vh' }}>
              <h2 style={{
                fontSize: '2rem',
                color: colors.text,
                fontFamily: 'Georgia, "Times New Roman", serif',
                fontWeight: '400',
                margin: 0,
                marginBottom: '1vh'
              }}>
                Create Your Account
              </h2>
              <p style={{
                color: colors.text,
                opacity: 0.8,
                fontSize: '1rem',
                margin: 0
              }}>
                Join us to unlock your personalized financial plan
              </p>
            </div>

            {/* Sign Up Form */}
            <form onSubmit={handleSignUp} style={{ width: '100%' }}>
              {/* Email or Phone Input */}
              <div style={{ marginBottom: '3vh' }}>
                <label style={{
                  display: 'block',
                  color: colors.text,
                  fontSize: '1rem',
                  marginBottom: '1vh',
                  fontWeight: '500'
                }}>
                  Email or Phone Number
                </label>
                <input
                  type="text"
                  value={signUpData.emailOrPhone}
                  onChange={(e) => setSignUpData({ ...signUpData, emailOrPhone: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '1.5vh 1vw',
                    border: `2px solid ${getSignUpInputBorderColor(signUpErrors.emailOrPhone)}`,
                    borderRadius: '0.5rem',
                    backgroundColor: 'transparent',
                    color: colors.text,
                    fontSize: '1rem',
                    outline: 'none',
                    transition: 'border-color 0.3s ease',
                    fontFamily: 'Georgia, "Times New Roman", serif'
                  }}
                  placeholder="Enter your email or phone number"
                />
                {signUpErrors.emailOrPhone && (
                  <p style={{
                    color: '#ef4444',
                    fontSize: '0.875rem',
                    margin: '0.5vh 0 0 0'
                  }}>
                    Please enter a valid email or phone number
                  </p>
                )}
              </div>

              {/* Password Input */}
              <div style={{ marginBottom: '3vh' }}>
                <label style={{
                  display: 'block',
                  color: colors.text,
                  fontSize: '1rem',
                  marginBottom: '1vh',
                  fontWeight: '500'
                }}>
                  Password
                </label>
                <input
                  type="password"
                  value={signUpData.password}
                  onChange={(e) => setSignUpData({ ...signUpData, password: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '1.5vh 1vw',
                    border: `2px solid ${getSignUpInputBorderColor(signUpErrors.password)}`,
                    borderRadius: '0.5rem',
                    backgroundColor: 'transparent',
                    color: colors.text,
                    fontSize: '1rem',
                    outline: 'none',
                    transition: 'border-color 0.3s ease',
                    fontFamily: 'Georgia, "Times New Roman", serif'
                  }}
                  placeholder="Create a password (min 6 characters)"
                />
                {signUpErrors.password && (
                  <p style={{
                    color: '#ef4444',
                    fontSize: '0.875rem',
                    margin: '0.5vh 0 0 0'
                  }}>
                    Password must be at least 6 characters long
                  </p>
                )}
              </div>

              {/* Confirm Password Input */}
              <div style={{ marginBottom: '4vh' }}>
                <label style={{
                  display: 'block',
                  color: colors.text,
                  fontSize: '1rem',
                  marginBottom: '1vh',
                  fontWeight: '500'
                }}>
                  Confirm Password
                </label>
                <input
                  type="password"
                  value={signUpData.confirmPassword}
                  onChange={(e) => setSignUpData({ ...signUpData, confirmPassword: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '1.5vh 1vw',
                    border: `2px solid ${getSignUpInputBorderColor(signUpErrors.confirmPassword || signUpErrors.passwordMatch)}`,
                    borderRadius: '0.5rem',
                    backgroundColor: 'transparent',
                    color: colors.text,
                    fontSize: '1rem',
                    outline: 'none',
                    transition: 'border-color 0.3s ease',
                    fontFamily: 'Georgia, "Times New Roman", serif'
                  }}
                  placeholder="Confirm your password"
                />
                {(signUpErrors.confirmPassword || signUpErrors.passwordMatch) && (
                  <p style={{
                    color: '#ef4444',
                    fontSize: '0.875rem',
                    margin: '0.5vh 0 0 0'
                  }}>
                    {signUpErrors.confirmPassword ? 'Please confirm your password' : 'Passwords do not match'}
                  </p>
                )}
              </div>

              {/* Sign Up Button */}
              <button
                type="submit"
                style={{
                  width: '100%',
                  borderRadius: '9999px',
                  border: `1px solid ${isDarkMode ? colors.buttonBg : colors.text}`,
                  background: isDarkMode ? colors.buttonBg : 'transparent',
                  color: isDarkMode ? colors.buttonText : colors.text,
                  fontWeight: '600',
                  fontSize: '1.125rem',
                  height: '48px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  marginBottom: '3vh'
                }}
                onMouseEnter={(e) => {
                  if (isDarkMode) {
                    e.currentTarget.style.backgroundColor = colors.accent;
                    e.currentTarget.style.borderColor = colors.accent;
                  } else {
                    e.currentTarget.style.color = colors.accent;
                    e.currentTarget.style.borderColor = colors.accent;
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }
                }}
                onMouseLeave={(e) => {
                  if (isDarkMode) {
                    e.currentTarget.style.backgroundColor = colors.buttonBg;
                    e.currentTarget.style.borderColor = colors.buttonBg;
                  } else {
                    e.currentTarget.style.color = colors.text;
                    e.currentTarget.style.borderColor = colors.text;
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }
                }}
              >
                Create Account
              </button>

              {/* Divider */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                marginBottom: '3vh'
              }}>
                <div style={{
                  flex: 1,
                  height: '1px',
                  backgroundColor: colors.text,
                  opacity: 0.3
                }} />
                <span style={{
                  color: colors.text,
                  opacity: 0.7,
                  padding: '0 2vw',
                  fontSize: '0.875rem'
                }}>
                  or
                </span>
                <div style={{
                  flex: 1,
                  height: '1px',
                  backgroundColor: colors.text,
                  opacity: 0.3
                }} />
              </div>

              {/* Google Sign Up Button */}
              <button
                type="button"
                onClick={handleGoogleSignUp}
                style={{
                  width: '100%',
                  borderRadius: '9999px',
                  border: `1px solid ${colors.text}`,
                  background: 'transparent',
                  color: colors.text,
                  fontWeight: '600',
                  fontSize: '1.125rem',
                  height: '48px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '1vw'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                  <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                  <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                  <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                </svg>
                Continue with Google
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Login Modal */}
      {showLoginModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            opacity: isLoginModalClosing ? 0 : (isLoginModalOpening ? 0 : 1),
            transition: 'opacity 0.15s ease'
          }}
          onClick={closeLoginModal}
        >
          <div
            style={{
              backgroundColor: colors.background,
              borderRadius: '1rem',
              padding: '3vh 3vw',
              maxWidth: '400px',
              width: '90%',
              maxHeight: '90vh',
              overflowY: 'auto',
              position: 'relative',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
              border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.15)' : 'none'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={closeLoginModal}
              style={{
                position: 'absolute',
                top: '1.5vh',
                right: '1.5vw',
                background: 'transparent',
                border: 'none',
                color: colors.text,
                cursor: 'pointer',
                fontSize: '1.5rem',
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '50%',
                transition: 'background-color 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
              title="Close"
            >
              ×
            </button>

            {/* Modal Content */}
            <div style={{ textAlign: 'center' }}>
              <h2 style={{
                color: colors.text,
                marginBottom: '3vh',
                fontSize: '1.75rem',
                fontWeight: '600'
              }}>
                Welcome
              </h2>

              <form onSubmit={handleLogin} style={{ width: '100%' }}>
                {/* Email Input */}
                <div style={{ marginBottom: '2vh', textAlign: 'left' }}>
                  <label style={{
                    display: 'block',
                    marginBottom: '0.5vh',
                    color: colors.text,
                    fontSize: '0.875rem',
                    fontWeight: '500'
                  }}>
                    Email
                  </label>
                  <input
                    type="email"
                    value={loginData.email}
                    onChange={(e) => setLoginData({ ...loginData, email: e.target.value })}
                    className={loginErrors.email ? 'error' : ''}
                    style={{
                      width: '100%',
                      padding: '1vh 1vw',
                      border: `2px solid ${getLoginInputBorderColor(loginErrors.email)}`,
                      borderRadius: '0.5rem',
                      backgroundColor: 'transparent',
                      color: colors.text,
                      fontSize: '1rem',
                      outline: 'none',
                      transition: 'border-color 0.3s ease',
                      boxSizing: 'border-box'
                    }}
                    placeholder="Enter your email"
                  />
                  {loginErrors.email && (
                    <p style={{
                      color: '#ef4444',
                      fontSize: '0.75rem',
                      marginTop: '0.5vh',
                      margin: '0.5vh 0 0 0'
                    }}>
                      Please enter a valid email address
                    </p>
                  )}
                </div>

                {/* Password Input */}
                <div style={{ marginBottom: '3vh', textAlign: 'left' }}>
                  <label style={{
                    display: 'block',
                    marginBottom: '0.5vh',
                    color: colors.text,
                    fontSize: '0.875rem',
                    fontWeight: '500'
                  }}>
                    Password
                  </label>
                  <input
                    type="password"
                    value={loginData.password}
                    onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                    className={loginErrors.password ? 'error' : ''}
                    style={{
                      width: '100%',
                      padding: '1vh 1vw',
                      border: `2px solid ${getLoginInputBorderColor(loginErrors.password)}`,
                      borderRadius: '0.5rem',
                      backgroundColor: 'transparent',
                      color: colors.text,
                      fontSize: '1rem',
                      outline: 'none',
                      transition: 'border-color 0.3s ease',
                      boxSizing: 'border-box'
                    }}
                    placeholder="Enter your password"
                  />
                  {loginErrors.password && (
                    <p style={{
                      color: '#ef4444',
                      fontSize: '0.75rem',
                      marginTop: '0.5vh',
                      margin: '0.5vh 0 0 0'
                    }}>
                      Password must be at least 6 characters
                    </p>
                  )}
                </div>

                {/* Login Button */}
                <button
                  type="submit"
                  style={{
                    width: '100%',
                    borderRadius: '9999px',
                    border: 'none',
                    background: colors.accent,
                    color: '#000',
                    fontWeight: '600',
                    fontSize: '1.125rem',
                    height: '48px',
                    cursor: 'pointer',
                    transition: 'background-color 0.3s ease',
                    marginBottom: '2vh'
                  }}
                  onMouseEnter={(e) => {
                    if (isDarkMode) {
                      e.currentTarget.style.backgroundColor = '#c49665';
                    } else {
                      e.currentTarget.style.backgroundColor = '#e5b284';
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = colors.accent;
                  }}
                >
                  Log In
                </button>

                {/* Divider */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '2vh'
                }}>
                  <div style={{
                    flex: 1,
                    height: '1px',
                    backgroundColor: colors.text,
                    opacity: 0.3
                  }} />
                  <span style={{
                    color: colors.text,
                    opacity: 0.7,
                    padding: '0 1vw',
                    fontSize: '0.875rem'
                  }}>
                    or
                  </span>
                  <div style={{
                    flex: 1,
                    height: '1px',
                    backgroundColor: colors.text,
                    opacity: 0.3
                  }} />
                </div>

                {/* Google Login Button */}
                <button
                  type="button"
                  onClick={handleGoogleLogin}
                  style={{
                    width: '100%',
                    borderRadius: '9999px',
                    border: `1px solid ${colors.text}`,
                    background: 'transparent',
                    color: colors.text,
                    fontWeight: '600',
                    fontSize: '1.125rem',
                    height: '48px',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                  </svg>
                  Continue with Google
                </button>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Add the universal dark mode toggle */}
      {DarkModeToggle}
    </div>
  );
}
