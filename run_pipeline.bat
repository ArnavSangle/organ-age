@echo off
REM ============================================================
REM  Organ-Age: Full Reproducibility Pipeline
REM  Runs every stage from raw data to paper figures and metrics.
REM
REM  Prerequisites:
REM    - Python 3.10+ with packages: torch, numpy, pandas, scipy,
REM      scikit-learn, umap-learn, matplotlib, seaborn
REM    - Raw datasets placed under data/raw/:
REM        data/raw/gtex/       (GTEx v10 bulk RNA-seq)
REM        data/raw/chexpert/   (CheXpert chest radiographs)
REM        data/raw/ixi/        (IXI T1-weighted MRI)
REM    - GPU recommended for training stages (use --device cpu to
REM      run on CPU, but expect longer runtimes)
REM
REM  Usage:
REM    run_pipeline.bat              Run full pipeline (GPU)
REM    run_pipeline.bat cpu          Run full pipeline (CPU)
REM    run_pipeline.bat --from 4     Resume from stage 4
REM ============================================================

setlocal enabledelayedexpansion

set DEVICE=auto
set FROM_STAGE=1

:parse_args
if "%~1"=="cpu"       ( set DEVICE=cpu& shift& goto parse_args )
if "%~1"=="--from"    ( set FROM_STAGE=%~2& shift& shift& goto parse_args )
if "%~1"=="--device"  ( set DEVICE=%~2& shift& shift& goto parse_args )

echo.
echo ============================================================
echo  Organ-Age Reproducibility Pipeline
echo  Device: %DEVICE%    Starting from stage: %FROM_STAGE%
echo ============================================================
echo.

REM ------- Stage 1: Train unimodal encoders -------------------
if %FROM_STAGE% LEQ 1 (
    echo [Stage 1/9] Training unimodal encoders ...
    python src/organon_multimodal/training/train_unimodal_encoders_v3.py ^
        --device %DEVICE%
    if errorlevel 1 ( echo FAILED at Stage 1 & exit /b 1 )
    echo [Stage 1/9] Done.
    echo.
)

REM ------- Stage 2: Contrastive alignment ---------------------
if %FROM_STAGE% LEQ 2 (
    echo [Stage 2/9] Training contrastive alignment ...
    python src/chimera_fusion/alignment/train_alignment_contrastive_v2.py ^
        --device %DEVICE%
    if errorlevel 1 ( echo FAILED at Stage 2 & exit /b 1 )
    echo [Stage 2/9] Done.
    echo.
)

REM ------- Stage 3: Cross-fusion transformer ------------------
if %FROM_STAGE% LEQ 3 (
    echo [Stage 3/9] Training cross-fusion transformer ...
    python src/vitalis_organage/v4_5_train_crossfusion.py ^
        --device %DEVICE% --epochs 40
    if errorlevel 1 ( echo FAILED at Stage 3 & exit /b 1 )
    echo [Stage 3/9] Done.
    echo.
)

REM ------- Stage 4: Build normative organ-age table -----------
if %FROM_STAGE% LEQ 4 (
    echo [Stage 4/9] Building organ-age normative table ...
    python src/vitalis_organage/build_organ_age_targets.py ^
        --device %DEVICE%
    if errorlevel 1 ( echo FAILED at Stage 4 & exit /b 1 )
    echo [Stage 4/9] Done.
    echo.
)

REM ------- Stage 5: Calibrate predictions ---------------------
if %FROM_STAGE% LEQ 5 (
    echo [Stage 5/9] Calibrating predictions ...
    python src/vitalis_organage/calibrate_v4.py
    if errorlevel 1 ( echo FAILED at Stage 5 & exit /b 1 )
    echo [Stage 5/9] Done.
    echo.
)

REM ------- Stage 6: Inference and reporting -------------------
if %FROM_STAGE% LEQ 6 (
    echo [Stage 6/9] Running inference and generating reports ...
    python src/vitalis_organage/v4_infer.py
    if errorlevel 1 ( echo FAILED at Stage 6 & exit /b 1 )
    echo [Stage 6/9] Done.
    echo.
)

REM ------- Stage 7: Explainability (IG + gene attribution) ----
if %FROM_STAGE% LEQ 7 (
    echo [Stage 7/9] Computing explainability analyses ...
    python src/vitalis_organage/v4_5_compute_latent_ig.py --device %DEVICE%
    if errorlevel 1 ( echo WARNING: latent IG failed, continuing... )
    python src/vitalis_organage/v4_5_ig_to_genes.py
    if errorlevel 1 ( echo WARNING: gene attribution failed, continuing... )
    echo [Stage 7/9] Done.
    echo.
)

REM ------- Stage 8: Compute metrics and statistical tests -----
if %FROM_STAGE% LEQ 8 (
    echo [Stage 8/9] Computing metrics and statistical tests ...
    python results/compute_all_metrics.py
    if errorlevel 1 ( echo FAILED at Stage 8 & exit /b 1 )
    python results/compute_statistical_tests.py
    if errorlevel 1 ( echo WARNING: statistical tests failed, continuing... )
    echo [Stage 8/9] Done.
    echo.
)

REM ------- Stage 9: Generate figures --------------------------
if %FROM_STAGE% LEQ 9 (
    echo [Stage 9/9] Generating figures ...
    python src/vitalis_organage/v4_visualize.py
    if errorlevel 1 ( echo WARNING: figure generation failed, continuing... )
    echo [Stage 9/9] Done.
    echo.
)

echo ============================================================
echo  Pipeline complete.
echo  Outputs:
echo    data/analysis/organ_age_normative.parquet
echo    data/analysis/organ_age_calibrated.parquet
echo    results/ablation_metrics.csv
echo    results/statistical_tests.json
echo    figures/paper/
echo ============================================================
