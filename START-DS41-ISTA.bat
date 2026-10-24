@echo off
rem START-DS41-ISTA.bat - serve the ISTA-DASLab DeepSeek-V4.1-Flash IQ2_XXS quant (138 GB, 4 split files) on Windows,
rem the same API at http://127.0.0.1:8150/v1. Config: tools\v41\serve\strata-ds41-ista-windows.json.
rem Against START-DS41.bat (our GSQ model) on this PC, 2026-10-24: 30-45% faster decode, 20-25% slower prefill on 8K-32K
rem prompts, ~2% higher ppl, loads in ~54 s, NO image input (the files have no exp_probs_b_vl vision tensors).
rem It uses our Engram tables, rank.bin and DSpark sidecar from models\1017. Stop the other server first (same port).
cd /d "%~dp0"
call START-DS41.bat --config tools\v41\serve\strata-ds41-ista-windows.json --port 8150 %*
