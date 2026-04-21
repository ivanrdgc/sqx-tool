@echo off
set PY_HOME=%~dp0src\python
set PYTHONPATH=%~dp0src;%PYTHONPATH%
set JUPYTER_CONFIG_DIR=%~dp0src\jupyter\config
set JUPYTER_DATA_DIR=%~dp0src\jupyter\data
set JUPYTER_RUNTIME_DIR=%~dp0src\jupyter\runtime
set IPYTHONDIR=%~dp0src\jupyter\ipython
"%PY_HOME%\python.exe" -m notebook --notebook-dir="%~dp0Notebooks" %*
