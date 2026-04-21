@echo off
set PY_HOME=%~dp0src\python
"%PY_HOME%\python.exe" -m notebook --notebook-dir="%~dp0Notebooks" %*
