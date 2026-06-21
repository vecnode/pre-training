@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"

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
echo Output folder will be created automatically as:
echo   [dataset_name]_PNG
echo If it already exists, conversion will resume.
echo.

call "%SCRIPT_DIR%uv_bootstrap.bat"
if errorlevel 1 goto :end

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%convert_pdf_to_png.ps1" -DatasetPath "%DATASET_PATH%" -PythonExe "%UFO_PYTHON%"

echo.
echo Converter finished. Review conversion_log.txt for details.

:end
echo.
pause
