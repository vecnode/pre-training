@echo off
rem -----------------------------------------------------------------------------
rem Summarize OCR CSV rows using Ollama and write DATASET_SUMMARIES.csv.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

rem Resolve paths.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
set "OUT_DIR=%ROOT_DIR%\output"

title OCR CSV Summaries with Ollama
echo.
echo ================================================
echo   OCR CSV Summaries (Ollama HTTP API)
echo ================================================
echo.
echo Enter OCR CSV path or dataset base name.
echo Examples: Release_1_OCR.csv  or  Release_1  or  C:\data\Release_1_OCR.csv
echo.

set /p "OCR_INPUT=OCR CSV input: "
if "%OCR_INPUT%"=="" (
	echo.
	echo No input provided. Exiting.
	goto :end
)

set "OCR_FILE="
rem Resolve OCR CSV input path from direct/file/base-name options.
if exist "%OCR_INPUT%" (
	set "OCR_FILE=%OCR_INPUT%"
) else if exist "%OUT_DIR%\%OCR_INPUT%" (
	set "OCR_FILE=%OUT_DIR%\%OCR_INPUT%"
) else if exist "%OUT_DIR%\%OCR_INPUT%_OCR.csv" (
	set "OCR_FILE=%OUT_DIR%\%OCR_INPUT%_OCR.csv"
) else if exist "%OUT_DIR%\%OCR_INPUT%.csv" (
	set "OCR_FILE=%OUT_DIR%\%OCR_INPUT%.csv"
)

if not exist "%OCR_FILE%" (
	echo.
	echo OCR CSV not found:
	echo   %OCR_FILE%
	goto :end
)

rem Derive output file name from OCR input file.
for %%I in ("%OCR_FILE%") do set "OCR_BASENAME=%%~nI"
set "DATASET_NAME=%OCR_BASENAME%"
if /I "%DATASET_NAME:~-4%"=="_OCR" set "DATASET_NAME=%DATASET_NAME:~0,-4%"
if "%DATASET_NAME%"=="" set "DATASET_NAME=dataset"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set "OUT_FILE=%OUT_DIR%\%DATASET_NAME%_SUMMARIES.csv"

echo.
set /p "OLLAMA_MODEL=Ollama model (blank = first available): "

echo.
echo Input : %OCR_FILE%
echo Output: %OUT_FILE%
echo.

rem Execute summarization, optionally with explicit model.
if defined OLLAMA_MODEL (
	"%UFO_PYTHON%" "%SCRIPT_DIR%summarize_ocr_ollama.py" --input "%OCR_FILE%" --output "%OUT_FILE%" --model "%OLLAMA_MODEL%"
) else (
	"%UFO_PYTHON%" "%SCRIPT_DIR%summarize_ocr_ollama.py" --input "%OCR_FILE%" --output "%OUT_FILE%"
)

echo.
echo Summary run finished.

:end
echo.
pause
