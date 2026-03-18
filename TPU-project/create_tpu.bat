@echo off
chcp 65001 >nul 2>&1

:: TRC TPU VM 创建脚本 -- 依次尝试，成功即停止
:: 用法: create_tpu.bat
:: 实测: 只有 v4 有 on-demand 权限，v5e/v6e 只有 spot
:: 优先级: on-demand v4 > spot v4 > spot v5e > spot v6e

set VM_NAME=tpu-dev

echo ==========================================
echo  TRC TPU VM 自动创建
echo  策略: on-demand v4 优先，fallback spot
echo ==========================================

:: ====== ON-DEMAND (只有 v4 有权限) ======

echo.
echo --- [1/7] v4-8 @ us-central2-b (on-demand)...
call gcloud compute tpus tpu-vm create %VM_NAME% --zone=us-central2-b --accelerator-type=v4-8 --version=tpu-ubuntu2204-base
if %errorlevel% equ 0 ( set S_ZONE=us-central2-b& set S_TYPE=v4-8& set S_MODE=on-demand& goto :done )
echo     FAILED

echo.
echo ==========================================
echo  on-demand 没容量，开始尝试 spot...
echo ==========================================

:: ====== SPOT FALLBACK ======

echo.
echo --- [2/7] v4-8 @ us-central2-b (spot)...
call gcloud compute tpus tpu-vm create %VM_NAME% --zone=us-central2-b --accelerator-type=v4-8 --version=tpu-ubuntu2204-base --spot
if %errorlevel% equ 0 ( set S_ZONE=us-central2-b& set S_TYPE=v4-8& set S_MODE=spot& goto :done )
echo     FAILED

echo.
echo --- [3/7] v5litepod-1 @ us-central1-a (spot)...
call gcloud compute tpus tpu-vm create %VM_NAME% --zone=us-central1-a --accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite --spot
if %errorlevel% equ 0 ( set S_ZONE=us-central1-a& set S_TYPE=v5litepod-1& set S_MODE=spot& goto :done )
echo     FAILED

echo.
echo --- [4/7] v5litepod-1 @ europe-west4-b (spot)...
call gcloud compute tpus tpu-vm create %VM_NAME% --zone=europe-west4-b --accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite --spot
if %errorlevel% equ 0 ( set S_ZONE=europe-west4-b& set S_TYPE=v5litepod-1& set S_MODE=spot& goto :done )
echo     FAILED

echo.
echo --- [5/7] v6e-1 @ europe-west4-a (spot)...
call gcloud compute tpus tpu-vm create %VM_NAME% --zone=europe-west4-a --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e --spot
if %errorlevel% equ 0 ( set S_ZONE=europe-west4-a& set S_TYPE=v6e-1& set S_MODE=spot& goto :done )
echo     FAILED

echo.
echo --- [6/7] v6e-1 @ us-east1-d (spot)...
call gcloud compute tpus tpu-vm create %VM_NAME% --zone=us-east1-d --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e --spot
if %errorlevel% equ 0 ( set S_ZONE=us-east1-d& set S_TYPE=v6e-1& set S_MODE=spot& goto :done )
echo     FAILED

:: ====== 最后再试一次 on-demand v4 ======

echo.
echo --- [7/7] v4-8 @ us-central2-b (on-demand, retry)...
call gcloud compute tpus tpu-vm create %VM_NAME% --zone=us-central2-b --accelerator-type=v4-8 --version=tpu-ubuntu2204-base
if %errorlevel% equ 0 ( set S_ZONE=us-central2-b& set S_TYPE=v4-8& set S_MODE=on-demand& goto :done )

echo.
echo ==========================================
echo  ALL FAILED - on-demand 和 spot 都没容量
echo  稍后再试，或先用 Colab 免费 TPU
echo ==========================================
exit /b 1

:done
echo.
echo ==========================================
echo  SUCCESS! TPU VM created
echo     Name: %VM_NAME%
echo     Zone: %S_ZONE%
echo     Type: %S_TYPE%
echo     Mode: %S_MODE%
echo.
echo  Next:
echo     SSH:  gcloud compute tpus tpu-vm ssh %VM_NAME% --zone=%S_ZONE%
if "%S_MODE%"=="on-demand" (
echo     Stop: gcloud compute tpus tpu-vm stop %VM_NAME% --zone=%S_ZONE%
echo     Start:gcloud compute tpus tpu-vm start %VM_NAME% --zone=%S_ZONE%
) else (
echo     [spot] 不支持 stop，结束后直接 delete
)
echo     Delete:gcloud compute tpus tpu-vm delete %VM_NAME% --zone=%S_ZONE% --quiet
echo ==========================================
exit /b 0
