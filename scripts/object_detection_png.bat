@echo off
rem -----------------------------------------------------------------------------
rem Run YOLO object detection over PNG pages and write DATASET_OBJS.csv.
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
setlocal EnableExtensions

rem Resolve paths.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
set "OUT_DIR=%ROOT_DIR%\output"

title YOLO Object Detection on PNG Dataset
echo.
echo ======================================================
echo   YOLO Detection (PNG -^> output\DATASET_OBJS.csv)
echo ======================================================
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

rem Resolve dataset input to a PNG folder path: exact path, path under the
rem project root, legacy "<name>_PNG" convention, or newest matching
rem outputs\<timestamp>_<slug> folder from convert_pdf_to_png.ps1.
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%resolve_png_dir.ps1" -DatasetInput "%DATASET_INPUT%" -ProjectRoot "%ROOT_DIR%"`) do set "PNG_DIR=%%P"

if not exist "%PNG_DIR%" (
	echo.
	echo PNG dataset folder not found for input: "%DATASET_INPUT%"
	echo Looked for: an exact path, a path under the project root, the legacy
	echo "<name>_PNG" convention, and outputs\[timestamp]_[slug] folders.
	echo Run convert_pdf_to_png.bat / exec_1.bat first.
	goto :end
)

rem Derive output file name from dataset folder.
for %%I in ("%PNG_DIR%") do set "DATASET_FOLDER=%%~nxI"
set "DATASET_NAME=%DATASET_FOLDER%"
if /I "%DATASET_NAME:~-4%"=="_PNG" set "DATASET_NAME=%DATASET_NAME:~0,-4%"
if "%DATASET_NAME%"=="" set "DATASET_NAME=dataset"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set "OUT_FILE=%OUT_DIR%\%DATASET_NAME%_OBJS.csv"

echo.
echo Input : %PNG_DIR%
echo Output: %OUT_FILE%
echo.

rem Execute YOLO pipeline.
"%UFO_PYTHON%" "%SCRIPT_DIR%object_detection_png.py" --image-dir "%PNG_DIR%" --output "%OUT_FILE%"

echo.
echo YOLO run finished.

:end
echo.
pause
