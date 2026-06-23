@echo off
rem -----------------------------------------------------------------------------
rem Convert dataset PDFs to PNG with resume support.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

rem Resolve paths.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

title PDF to PNG Resume Converter
echo.
echo ================================================
echo   PDF to PNG Converter (Resume Supported)
echo ================================================
echo.
echo Enter the dataset folder path that contains PDF files.
echo Examples: Release_1   or   C:\data\my_dataset
echo.

set /p "DATASET_PATH=Dataset path: "
if "%DATASET_PATH%"=="" (
	echo.
	echo No dataset path provided. Exiting.
	goto :end
)

echo.
rem Execute converter PowerShell wrapper.
echo Output folder will be created automatically as:
echo   [dataset_name]_PNG
echo If it already exists, conversion will resume.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%convert_pdf_to_png.ps1" -DatasetPath "%DATASET_PATH%" -PythonExe "%UFO_PYTHON%"

echo.
echo Converter finished. Review conversion_log.txt for details.

:end
echo.
pause
