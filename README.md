# Quantitative Microstructure Research Archive

An organized collection of experiments in order-book signals, regime conditioning, local price dynamics and execution modeling.

## Research themes

- Order-flow and microstructure features across equities and digital assets.
- Regime-conditioned directional and market-making models.
- Kinematic versus autoregressive price representations.
- Backtesting, fill simulation and experiment diagnostics.

[Search the experiment inventory](docs/explorer.html) — download and open the standalone HTML catalog.

## Project entry points

| Project | Research focus |
|---|---|
| [L2 direction prediction](https://github.com/shah1403/l2-regime-research) | Order-book states, session-specific directional hit rates, timing and fee sensitivity |
| [Kalshi microstructure](https://github.com/shah1403/kalshi-microstructure) | Conditional next-taker-side prediction and joint fill risk |
| [Causal first passage](https://github.com/shah1403/first-passage-research) | Rolling distributions, label maturity and execution accounting |
| [Physics in Economics](https://github.com/shah1403/physics-in-economics) | Continuous double auctions and market-convergence dynamics |

## Archive guide

historical/ contains FI-2010 experiments, volatile-market studies, regime/trader scripts, earlier backtesters, local fill/equity outputs and kinematic-vs-AR analysis. data/market_data_2026-09-19.json.gz retains the supplied market-data archive. verification/source_inventory.csv indexes the files; MANIFEST.json records their hashes.

The archive is a source catalog. Its historical experiments have been indexed and syntax-inspected; independent replay coverage is documented in the project repositories above. Historical outputs retain their original experiment assumptions. Dataset-to-result mapping and executable-price interpretation require each experiment's configuration; FI-2010 transformed observations require additional native price/timestamp metadata for execution analysis. No aggregate performance statistic is asserted for this catalog.

## Large inputs

Run python restore_large_files.py to reconstruct inputs stored as numbered byte parts and verify their SHA-256 against large_files.json.

Original files and source-specific attribution are retained. No new license is imposed on third-party data or source material.

By Aarush Shah.
