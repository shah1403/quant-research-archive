# Additional quantitative research archive

Recovered experimental code, diagnostics, and historical outputs that are outside the fully replayed core projects.

`historical/` includes FI-2010 microstructure, volatile-market experiments, regime/trader scripts, earlier backtesters, local fills/equity, and kinematic-vs-AR analysis. `data/market_data_2026-09-19.json.gz` is the original recovered data archive; its relationship to each result is not independently established.

**Status: recovered and indexed, not fully reproduced.** These files are preserved in their original form. Do not interpret bundled plots or old summary statistics as verified performance. Missing dependencies, missing source datasets, unsupported fee assumptions, and historical look-ahead may apply. The audited first-passage and Kalshi projects provide the verified entry points.

## Evidence gaps

- The exact Raedan forward-test ledger supporting >60% accuracy and per-trade Sharpe about 0.5 was not recovered.
- The original May 2021 market-making inputs/configuration supporting $36,503 / Sharpe 15.58 and $19,664 / Sharpe 10.69 were not recovered.
- The complete July minute-bar / Nifty and EURUSD project dataset-to-code chain was not recovered.
- FI-2010 transformed observations alone do not establish native prices, instrument identities, or timestamps sufficient for executable P&L.

This archive is intended for private preservation and follow-up. No new open-source license is asserted for third-party data or unidentified third-party strategy code. Source-specific attribution remains in original files. `MANIFEST.json` records exact file hashes.

## Large input files

Original inputs above GitHub’s single-file limit are included as numbered byte parts. Run `python restore_large_files.py` once to reconstruct them and verify their SHA-256. The manifest is `large_files.json`; no data is discarded.

## Source explorer

[Search the experiment inventory](docs/explorer.html) — download and open the standalone HTML file. This is a provenance catalog, not a performance presentation.
