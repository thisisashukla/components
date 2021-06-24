@echo off

set envname=%1
set projname=%2

REM ECHO %envname%
REM ECHO %projname%

call :ActivateEnv %envname%
call :GotoProject %projname%

EXIT /B %ERRORLEVEL%

:ActivateEnv
call E:
call cd Envs
call cd %~1
call cd Scripts
call activate.bat
REM cmd /c "activate.bat"

EXIT /B 0

:GotoProject
call cd E:\Github
call cd %~1

EXIT /B 0