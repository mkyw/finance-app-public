// Static client-side US city autocomplete.
//
// city_list.json is emitted by pipeline/export/export_city_puma_map.R
// (section 9) — incorporated places + CDPs joined with 2020 Decennial
// P1 populations, sorted by population descending, PR excluded.
//
// The list ships as a static asset at /city_list.json (public/), NOT
// bundled into the JS. Call initCityList() once on mount to populate
// the module-level cache; subsequent searchCities() calls are
// synchronous and zero-allocation against that cache.

export interface CityEntry {
  name: string;
  state: string;
  population: number;
}

export interface CityOption {
  label: string; // "Los Angeles, CA"
  name: string;
  state: string;
  population: number;
}

let CITIES: CityEntry[] = [];
let initPromise: Promise<void> | null = null;

export async function initCityList(): Promise<void> {
  if (CITIES.length > 0) return;
  // Deduplicate concurrent initializations: multiple effects/mounts that
  // hit this before the first fetch completes all await the same Promise.
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const res = await fetch("/city_list.json");
    CITIES = (await res.json()) as CityEntry[];
  })();
  try {
    await initPromise;
  } finally {
    initPromise = null;
  }
}

export function searchCities(query: string, limit: number = 8): CityOption[] {
  if (query.trim().length < 2) return [];
  const q = query.toLowerCase().trim();

  // startsWith (not includes) so "Los" surfaces Los Angeles first instead of
  // any name containing "los". CITIES is already ordered by population desc
  // so natural iteration yields populous matches first.
  const out: CityOption[] = [];
  for (const c of CITIES) {
    const name = c.name.toLowerCase();
    if (name.startsWith(q) || `${name}, ${c.state.toLowerCase()}`.startsWith(q)) {
      out.push({
        label: `${c.name}, ${c.state}`,
        name: c.name,
        state: c.state,
        population: c.population,
      });
      if (out.length >= limit) break;
    }
  }
  return out;
}
