@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
set "OUT_DIR=%ROOT_DIR%\output"

title OCR Detection on PNG Dataset
echo.
echo ================================================
echo   OCR Detection (PNG -> output\DATASET_OCR.md)
echo ================================================
echo.
echo Enter dataset base name OR PNG folder path.
echo Examples: Release_1  or  Release_1_PNG  or  C:\data\Release_1_PNG
echo.

set /p "DATASET_INPUT=Dataset: "
if "%DATASET_INPUT%"=="" (
	echo.
	echo No dataset input provided. Exiting.
	goto :end
)

set "PNG_DIR="

if exist "%DATASET_INPUT%" (
	if /I "%DATASET_INPUT:~-4%"=="_PNG" (
		set "PNG_DIR=%DATASET_INPUT%"
	) else if exist "%DATASET_INPUT%_PNG" (
		set "PNG_DIR=%DATASET_INPUT%_PNG"
	) else (
		set "PNG_DIR=%DATASET_INPUT%"
	)
) else if exist "%ROOT_DIR%\%DATASET_INPUT%" (
	if /I "%DATASET_INPUT:~-4%"=="_PNG" (
		set "PNG_DIR=%ROOT_DIR%\%DATASET_INPUT%"
	) else if exist "%ROOT_DIR%\%DATASET_INPUT%_PNG" (
		set "PNG_DIR=%ROOT_DIR%\%DATASET_INPUT%_PNG"
	) else (
		set "PNG_DIR=%ROOT_DIR%\%DATASET_INPUT%"
	)
) else if exist "%ROOT_DIR%\%DATASET_INPUT%_PNG" (
	set "PNG_DIR=%ROOT_DIR%\%DATASET_INPUT%_PNG"
)

if not exist "%PNG_DIR%" (
	echo.
	echo PNG dataset folder not found:
	echo   %PNG_DIR%
	echo Run convert_pdf_to_png.bat first.
	goto :end
)

for %%I in ("%PNG_DIR%") do set "DATASET_FOLDER=%%~nxI"
set "DATASET_NAME=%DATASET_FOLDER%"
if /I "%DATASET_NAME:~-4%"=="_PNG" set "DATASET_NAME=%DATASET_NAME:~0,-4%"
if "%DATASET_NAME%"=="" set "DATASET_NAME=dataset"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set "OUT_FILE=%OUT_DIR%\%DATASET_NAME%_OCR.md"

echo.
echo Input : %PNG_DIR%
echo Output: %OUT_FILE%
echo.

"%UFO_PYTHON%" "%SCRIPT_DIR%ocr_detection_png.py" --image-dir "%PNG_DIR%" --output "%OUT_FILE%"

echo.
echo OCR run finished.

:end
echo.
pause
