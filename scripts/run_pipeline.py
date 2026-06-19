"""Orchestrate the offline R pipeline end-to-end.

Runs (in order):
  1. Rscript pipeline/export/export_population.R
  2. Rscript pipeline/export/export_coefficients.R
  3. Sanity checks on pipeline/artifacts/ (shape + category coverage).

Working directory for each R step must be pipeline/fusionData/ (see
each R file's header comment).
"""


def main() -> None:
    raise NotImplementedError("Implement once the R export scripts land.")


if __name__ == "__main__":
    main()
