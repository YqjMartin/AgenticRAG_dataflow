
# 确保脚本以 UTF-8 with BOM 编码保存！
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 生成时间戳
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logfile = "output_${timestamp}.log"

Write-Host "正在运行 mygen.py，日志将保存到: $logfile`n"
python mygen.py | Tee-Object -FilePath $logfile
Write-Host "`n运行完成！日志已保存到: $logfile"