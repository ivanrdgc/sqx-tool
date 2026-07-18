@echo off
set PY_HOME=%~dp0src\python
"%PY_HOME%\python.exe" "%~dp0src\update_instruments_db.py" %*
