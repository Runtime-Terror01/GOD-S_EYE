# Get full path of the current project
$projectPath = (Get-Location).Path

# Define commands for each backend
$dashboardCmd = "Set-Location `"$projectPath\backend`"; & `"$projectPath\.venv\Scripts\Activate.ps1`"; python backend.py"
$comfyCmd     = "Set-Location `"$projectPath`"; & `"$projectPath\.venv\Scripts\Activate.ps1`"; python main.py"

# Start both processes safely (apostrophes, spaces, etc. handled)
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $dashboardCmd
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $comfyCmd
