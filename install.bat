REM filepath: c:\Projetos\Python\TransformersAI\install.bat
@echo off
echo Criando ambiente virtual...
python -m venv venv

echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo Atualizando pip...
pip install --upgrade pip

echo Instalando PyTorch primeiro...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo Instalando outras dependências...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ERRO: Falha na instalação das dependências!
    echo Tentando instalar dependências individuais...
    pip install flask==2.3.3
    pip install transformers
    pip install datasets
    pip install accelerate
    pip install beautifulsoup4
    pip install requests
    pip install python-dotenv
    pip install gradio
    pip install numpy
    pip install pandas
    pip install scikit-learn
    pip install email-validator
)

echo.
echo Verificando instalação...
python -c "import torch; print('PyTorch instalado:', torch.__version__)"
python -c "import transformers; print('Transformers instalado:', transformers.__version__)"
python -c "import flask; print('Flask instalado:', flask.__version__)"

echo.
echo Instalação concluída!
echo Para usar o sistema:
echo 1. Execute: venv\Scripts\activate.bat
echo 2. Execute: python app.py
echo 3. Acesse: http://localhost:5000
pause