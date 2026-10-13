# Compton scattering experiment (Phys 871, Sept–Oct 2026)

| Folder / file                                                 | Contents                                                                    |
|---------------------------------------------------------------|-----------------------------------------------------------------------------|
| `paper/compton_scattering_report.pdf`                         | **The report** (Physical Review style, REVTeX 4.2)                          |
| `paper/compton_scattering_report.tex`, `paper/references.bib` | Report source (LaTeX template) to edit                                      |
| `paper/figures/`, `paper/generated/`                          | Figures, tables and numbers produced by the analysis (do not edit by hand)  |
| `analysis/`                                                   | Analysis code, results tables and figures; see `analysis/README.md`         |
| `data/`                                                       | Raw spectra (`.mca`); see `data/README.md`                                  |
| `p471lab2.pdf`                                                | Laboratory instructions                                                     |
| `isotope_spectra.pdf`, `amptek.pdf`                           | Reference spectra and detector/MCA documentation                            |
| `Experiments in Modern Physics … .pdf`                        | Textbook (Melissinos & Napolitano, 2nd ed.)                                 |

## Requirements

- Python 3 with NumPy, SciPy and Matplotlib: `pip install -r requirements.txt`
- A LaTeX distribution with REVTeX 4.2 (TeX Live, MiKTeX or Tectonic), or Overleaf

## Rebuilding the report

1. Run the analysis (about 6 minutes). It regenerates all figures, tables and numbers.
   ```
   cd analysis
   python run_all.py
   ```
2. Compile the report.
   ```
   cd ../paper
   tectonic compton_scattering_report.tex
   ```
   `pdflatex` + `bibtex` (or `latexmk -pdf`) and Overleaf also work. On Overleaf, upload the whole `paper/` folder.

When edit the report, just change the text in `compton_scattering_report.tex`. The numbers come from `generated/macros.tex` (for example `\mFix`, `\alphaMean`), so they stay consistent with the analysis when it is rerun.
