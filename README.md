# finance-app

A household finance microsimulation engine that turns a few household facts —
age, income, location, housing tenure, household size — into a personalized picture of
where the money goes: a category-by-category spending breakdown, a gross→take-home tax
wedge, a four-way split of every dollar (committed outflows, debt service,
spending, savings), benefits eligibility screening, and a balance-sheet view.

> **Note — code showcase.** This repository is the application and serving-layer
> source. The offline data pipeline and its build artifacts (a large synthetic
> population and derived statistical models) are maintained separately and are
> **not included here**, so the app is published for reading rather than for
> turnkey local execution. The architecture below explains where that data layer
> plugs in.

## Website Flow
<p align="center">
  <img width="1440" height="900" alt="Landing Page" src="https://github.com/user-attachments/assets/69b38ebd-d4fe-41b3-81a6-042564796017" />
  <br /><em>Landing Page</em>
</p>

<p align="center">
  <img width="1440" height="900" alt="User Registration (1)" src="https://github.com/user-attachments/assets/3e4c43df-802e-4ee0-92eb-9641cab629e8" />
  <br /><em>User Registration — Step 1</em>
</p>

<p align="center">
  <img width="1440" height="900" alt="User Registration (2)" src="https://github.com/user-attachments/assets/b71d1a00-3b60-40eb-90b2-58d01a113393" />
  <br /><em>User Registration — Step 2</em>
</p>

<p align="center">
  <img width="1440" height="900" alt="Loading Screen" src="https://github.com/user-attachments/assets/c4adeb40-ee2e-4690-bfbd-1efc1a72c94e" />
  <br /><em>Loading Screen</em>
</p>

<p align="center">
  <img width="1440" height="900" alt="Spending Breakdown" src="https://github.com/user-attachments/assets/306b5a76-1735-4258-9783-a027b3738ef7" />
  <br /><em>Spending Breakdown</em>
</p>

<p align="center">
  <img width="1440" height="900" alt="Taxes Screen" src="https://github.com/user-attachments/assets/03736f44-5c84-42bb-8ad5-7d96d91f1643" />
  <br /><em>Taxes Screen</em>
</p>

## Architecture

Three subsystems with a strict **one-way data flow** — an offline pipeline
produces artifacts; a pure-Python serving layer reads them; a web app renders the
results:

```
data pipeline  →  artifacts (parquet + JSON)  →  models/  →  apps/api  →  apps/web
  (offline,            (the data layer,         (pure Python   (Django +    (Next.js)
   not in this repo)    not in this repo)        serving)        DRF)
```

- **`models/`** — the pure-Python serving layer (no web framework, no I/O beyond
  reading artifacts). It matches a household to a comparable cohort, derives
  per-category spending distributions, runs the allocation/optimization model,
  computes taxes, screens benefits, and assembles the balance sheet. This is the
  analytical core.
- **`apps/api/`** — Django 4.2 + Django REST Framework. Thin views over a
  `services.py` layer that orchestrates `models/`; views never import `models/`
  directly. The live surface analyzes a household profile and resolves a city to
  its statistical geography.
- **`apps/web/`** — Next.js 15 + React 19 + Tailwind 4 front end that collects
  the profile and renders the results.
- **`shared/`** — contract types and constants shared by `models/` and the API.

## Repository layout

```
apps/web/     Next.js front end
apps/api/     Django + DRF API (only the profiles app has a live view layer)
models/       pure-Python serving layer (matching, tax, optimizer, benefits, …)
shared/       contract types + constants
scripts/      maintenance / CI helper scripts
tests/        pytest suite (model + API layers)
```

## Tech stack

- **Backend:** Python 3.11+, Django 4.2, Django REST Framework, NumPy/SciPy,
  pandas + pyarrow, scikit-learn, statsmodels, and `taxcalc` for federal tax.
- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS 4.
- **Tooling:** pytest, ESLint, GitHub Actions CI (syntax/lint/build gates plus a
  full-history secret scan).

## Running locally

The API and model layer require the external data layer described above, so a
fresh clone will not produce live predictions. The pieces that do run standalone:

```bash
# Front end (UI, static):
cd apps/web && npm install && npm run dev      # http://localhost:3000

# Python syntax gate (no data required):
python3.11 -m compileall models shared apps/api
```

Backend configuration is read from environment variables (see `.env.example`);
copy it to `.env` and set at least `SECRET_KEY` before running Django.

## License

[MIT](./LICENSE).
