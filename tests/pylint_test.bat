ECHO OFF
ECHO Running Pylint Tests
Rem Hardcode your python path (for example if using venv)
set my_python=C:\Users\paedi\PycharmProjects\semesterproject_opti_para\venv\Scripts\python
Rem Or try finding your python
Rem FOR /F "tokens=* USEBACKQ" %%F IN (`where python`) DO (
Rem SET my_python=%%F
Rem )

Rem todo: install pip and pylint from batch
ECHO %my_python%
cd ..
for %%i in (*) do (
    ECHO %%i
    %my_python% -m pylint %%i --extension-pkg-whitelist=numpy --rcfile=tests/.pylintrc
)
cd tests