@echo off
setlocal

REM Build a portable WuziQ executable using Python 3.11 and PyInstaller.
py -3.11 -m pip install -U pip pyinstaller
py -3.11 -m PyInstaller -F -w -n WuziQ -p . src\ui_tk.py

endlocal
