#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="./Logs/output_${timestamp}.log"

echo "正在运行 mygen.py，日志将保存到: $logfile"
echo

python verify_current_benchmark.py | tee "$logfile"

sleep 5

echo
echo "运行完成！日志已保存到: $logfile"
