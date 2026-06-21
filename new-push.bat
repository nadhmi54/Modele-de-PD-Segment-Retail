@echo off
setlocal EnableExtensions

REM -----------------------------------------------------------------------------
REM new-push.bat - Create a new branch, commit all changes, and push to origin.
REM
REM Use when:  Starting a new feature/fix branch with your first commit.
REM Do NOT use: To sync develop with remote (use update.bat instead).
REM Remote:     origin
REM -----------------------------------------------------------------------------

set /p BRANCH="Enter new branch name: "
if not defined BRANCH (
    echo [ERROR] Branch name cannot be empty.
    endlocal & exit /b 1
)

echo [INFO] Creating branch '%BRANCH%'...
git checkout -b "%BRANCH%"
if errorlevel 1 (
    echo [ERROR] Failed to create branch '%BRANCH%'.
    endlocal & exit /b 1
)

echo [INFO] Staging all changes...
git add .
if errorlevel 1 (
    echo [ERROR] Failed to stage files.
    endlocal & exit /b 1
)

echo [INFO] Creating initial commit...
git commit -m "Initial commit on %BRANCH%"
if errorlevel 1 (
    echo [ERROR] Failed to commit. Nothing to commit or commit was rejected.
    endlocal & exit /b 1
)

echo [INFO] Pushing to origin/%BRANCH%...
git push origin "%BRANCH%"
if errorlevel 1 (
    echo [ERROR] Failed to push to origin/%BRANCH%.
    endlocal & exit /b 1
)

echo [SUCCESS] Branch '%BRANCH%' created and pushed to origin.
endlocal & exit /b 0