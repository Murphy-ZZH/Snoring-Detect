@echo off
setlocal enabledelayedexpansion

echo 正在将所有子目录中的文件移动到当前目录...

rem 遍历所有子目录中的文件
for /r %%f in (*) do (
    rem 获取文件名
    set filename=%%~nxf
    rem 移动文件到当前目录
    move "%%f" "%cd%\!filename!" >nul
)

echo 所有文件已移动到当前目录。
pause