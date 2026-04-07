$tasks = @('KBO_Predict_Day', 'KBO_Predict_Evening', 'KBO_Predict_Night')
foreach ($name in $tasks) {
    $task = Get-ScheduledTask -TaskName $name
    $task.Settings.WakeToRun = $true
    Set-ScheduledTask -TaskName $name -Settings $task.Settings
    Write-Host "[OK] $name WakeToRun 활성화 완료"
}
Get-ScheduledTask -TaskName 'KBO_Predict_*' | Format-Table TaskName, State -AutoSize
