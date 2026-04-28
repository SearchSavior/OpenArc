@echo off

echo OpenArc setup script for Windows

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

if not exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
    echo ERROR: Intel oneAPI not found.
    echo install from https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
    pause
    exit /b 1
)

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

uv sync
call .venv\Scripts\activate.bat

uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
uv pip install --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

set /p set_key="set OPENARC_API_KEY? (y/N): "
if /I not "%set_key%"=="y" goto :skip_key
set /p api_key="key (default: openarc-api-key): "
if "%api_key%"=="" set "api_key=openarc-api-key"
setx OPENARC_API_KEY "%api_key%"
:skip_key

openarc --help
pause