@echo off
set PY_HOME=%~dp0src\python
"%PY_HOME%\python.exe" -m jupyter notebook --notebook-dir="%~dp0Notebooks" %*
