@echo off
setlocal enabledelayedexpansion

echo ���ڽ�������Ŀ¼�е��ļ��ƶ�����ǰĿ¼...

rem ����������Ŀ¼�е��ļ�
for /r %%f in (*) do (
    rem ��ȡ�ļ���
    set filename=%%~nxf
    rem �ƶ��ļ�����ǰĿ¼
    move "%%f" "%cd%\!filename!" >nul
)

echo �����ļ����ƶ�����ǰĿ¼��
pause