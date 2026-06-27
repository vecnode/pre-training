@echo off
rem -----------------------------------------------------------------------------
rem Convenience entry point: delegate to the canonical deploy/deploy.bat, which
rem bootstraps the uv env and serves the LoRA adapter inference server.
rem Usage:  deploy.bat [host] [port]      (defaults: 127.0.0.1 8008)
rem Copyright (c) vecnode 2026
rem -----------------------------------------------------------------------------
call "%~dp0deploy\deploy.bat" %*
exit /b %errorlevel%
