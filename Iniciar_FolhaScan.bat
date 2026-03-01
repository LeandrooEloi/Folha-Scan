@echo off
cd /d "%~dp0"
echo Iniciando o FolhaScan...
echo Por favor, nao feche esta janela preta enquanto usar o app.
python -m streamlit run app.py
pause
