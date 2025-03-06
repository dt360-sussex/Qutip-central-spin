@echo off
echo Activating conda environment...
call conda activate dask_env
if errorlevel 1 (
    echo Failed to activate conda environment
    pause
    exit /b
)

echo Starting dask-worker...
dask-worker 192.168.1.101:8676 --nworkers=12
if errorlevel 1 (
    echo Failed to start dask-scheduler
    pause
    exit /b
)

pause