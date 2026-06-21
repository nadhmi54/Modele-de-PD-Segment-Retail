@echo off
setlocal EnableExtensions

REM -----------------------------------------------------------------------------
REM update.bat - Switch to develop and pull latest from origin.
REM
REM Use when:  Syncing your local develop branch before starting new work.
REM Do NOT use: To create/push a new feature branch (use new-push.bat instead).
REM Remote:     origin
REM Branch:     develop (integration branch for ContrAgri)
REM -----------------------------------------------------------------------------

git show-ref --verify --quiet refs/heads/develop
if errorlevel 1 (
    echo [ERROR] Branch 'develop' does not exist locally.
    echo [HINT]  Run: git fetch origin ^&^& git switch develop
    endlocal & exit /b 1
)

echo [INFO] Switching to 'develop'...
git switch develop
if errorlevel 1 (
    echo [ERROR] Failed to switch to 'develop'.
    endlocal & exit /b 1
)

echo [INFO] Pulling from origin/develop...
git pull origin develop
if errorlevel 1 (
    echo [ERROR] Failed to pull from origin/develop.
    endlocal & exit /b 1
)

echo [SUCCESS] 'develop' is up to date with origin.
endlocal & exit /b 0