# Commandes PLAN_100J — PowerShell (Windows)
param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$Py = "python"

switch ($Target) {
    "install" { & $Py -m pip install -r requirements.txt }
    "baseline" { & $Py baseline.py }
    "cv-rolling" { & $Py cv_rolling.py }
    "cv-group" { & $Py cv_group.py }
    "cv" { & $Py cv_rolling.py; & $Py cv_group.py }
    "train" { & $Py run_pipeline.py --final-only }
    "push" { & $Py optimize_push.py }
    "all" { & $Py baseline.py; & $Py cv_rolling.py; & $Py cv_group.py; & $Py run_pipeline.py --final-only }
    default {
        Write-Host "Usage: .\make.ps1 [install|baseline|cv-rolling|cv-group|cv|train|push|all]"
    }
}
