@echo off
setlocal enabledelayedexpansion

rem 初始化行和列计数器
set row=0
set col=0

rem 获取当前目录中的所有文件
for %%f in (*) do (
    rem 只处理文件，跳过子目录
    if not "%%f"=="%~nx0" (
        rem 构造新的文件名，格式为 0_0, 0_1, 0_2, ...
        set newname=!row!_!col!%%~xf
        rem 重命名文件
        ren "%%f" "!newname!"
        
        rem 更新列计数器
        set /a col+=1
        rem 如果列计数器达到10，换行并重置列计数器
        if !col! geq 10 (
            set col=0
            set /a row+=1
        )
    )
)

echo 文件已按 0_0, 0_1, ... 顺序重新命名。
pause
