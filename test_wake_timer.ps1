# Wake Timer 테스트 스크립트 (관리자 권한으로 실행 필요)
$testFile = "C:\Users\yhkan\Data_Analysis_Project\test_wake.txt"
$runAt = (Get-Date).AddMinutes(5).ToString('HH:mm')

$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
    -Argument "/c echo WakeTimer_Test_OK > `"$testFile`""

$trigger = New-ScheduledTaskTrigger -Once -At $runAt
$trigger.EndBoundary = (Get-Date).AddMinutes(7).ToString('s')

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -DeleteExpiredTaskAfter '00:02:00'

Register-ScheduledTask -TaskName 'KBO_WakeTest' `
    -Action $action -Trigger $trigger -Settings $settings `
    -RunLevel Highest -Force | Out-Null

Write-Host "[OK] Task registered: will run at $runAt"
Write-Host ""
Write-Host "===== Put PC to sleep now ====="
Write-Host "  Start menu -> Power -> Sleep"
Write-Host "  OR: enter 'y' below to sleep automatically in 30s"
Write-Host ""
Write-Host "PC should wake up at $runAt automatically."
Write-Host "Check result: $testFile will be created on success."
Write-Host ""

$choice = Read-Host "Auto-sleep in 30 seconds? (y/n)"
if ($choice -eq 'y') {
    Write-Host "Hibernating in 30 seconds... (PC will fully power down, then auto-wake)"
    Start-Sleep -Seconds 30
    shutdown.exe /h
}
