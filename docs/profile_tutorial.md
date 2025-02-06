### query, lock the gpu clock for reproducibility

Please check /scripts for detail.



### profile command in windows, I don't know why bat script failed.

When profile, if you can not lock the gpu clock(I can't do this on my laptop), kill unecessary processes that may
consume gpu resources.

```
setlocal
set OUTPUT_DIR=docs\profile_result\RTX4060
set EXECUTABLE_DIR=build\Release

ncu --set full -o "%OUTPUT_DIR%\00_cublas"              "%EXECUTABLE_DIR%\main.exe" 0
ncu --set full -o "%OUTPUT_DIR%\01_naive"               "%EXECUTABLE_DIR%\main.exe" 1
ncu --set full -o "%OUTPUT_DIR%\02_coalesce_memory"     "%EXECUTABLE_DIR%\main.exe" 2
ncu --set full -o "%OUTPUT_DIR%\03_tiled"               "%EXECUTABLE_DIR%\main.exe" 3
ncu --set full -o "%OUTPUT_DIR%\04_4x_coarsen_tiled"    "%EXECUTABLE_DIR%\main.exe" 4
ncu --set full -o "%OUTPUT_DIR%\05_8x_compute"          "%EXECUTABLE_DIR%\main.exe" 5
ncu --set full -o "%OUTPUT_DIR%\06_more_compute"        "%EXECUTABLE_DIR%\main.exe" 6
ncu --set full -o "%OUTPUT_DIR%\07_vectorize"           "%EXECUTABLE_DIR%\main.exe" 7
ncu --set full -o "%OUTPUT_DIR%\08_warp_tile"           "%EXECUTABLE_DIR%\main.exe" 8
ncu --set full -o "%OUTPUT_DIR%\09_double_buffering"           "%EXECUTABLE_DIR%\main.exe" 9

endlocal
```