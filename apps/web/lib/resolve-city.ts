// Backend-backed city -> PUMA resolution, given a CityOption selected from
// the static autocomplete list.

export interface CityOption {
  label: string;
  name: string;
  state: string;
}

export interface ResolvedCity {
  label: string;
  pumas: string[];
  resolved_via: "place" | "county";
  city: string;
  state_code: string;
  place_fips: string;
  county_fips: string;
}

export async function resolveCity(option: CityOption): Promise<ResolvedCity> {
  const res = await fetch("http://localhost:8000/api/profiles/resolve-city/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      city_name: option.name,
      county_name: "",
      state_code: option.state,
      addresstype: "city",
    }),
  });

  if (!res.ok) {
    throw new Error(`No coverage for ${option.label}`);
  }

  const data = (await res.json()) as {
    pumas: string[];
    resolved_via: "place" | "county";
    place_fips?: string;
    county_fips?: string;
  };

  return {
    label: option.label,
    pumas: data.pumas,
    resolved_via: data.resolved_via,
    city: option.name,
    state_code: option.state,
    place_fips: data.place_fips ?? "",
    county_fips: data.county_fips ?? "",
  };
}
