@echo off
setlocal enabledelayedexpansion

rem ��ʼ���к��м�����
set row=0
set col=0

rem ��ȡ��ǰĿ¼�е������ļ�
for %%f in (*) do (
    rem ֻ�����ļ���������Ŀ¼
    if not "%%f"=="%~nx0" (
        rem �����µ��ļ�������ʽΪ 0_0, 0_1, 0_2, ...
        set newname=!row!_!col!%%~xf
        rem �������ļ�
        ren "%%f" "!newname!"
        
        rem �����м�����
        set /a col+=1
        rem ����м������ﵽ10�����в������м�����
        if !col! geq 10 (
            set col=0
            set /a row+=1
        )
    )
)

echo �ļ��Ѱ� 0_0, 0_1, ... ˳������������
pause
