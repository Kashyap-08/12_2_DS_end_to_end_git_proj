echo [$(date)]: "START"

echo [$(date)]: "CREATING VENV WITH PYTHON 3.9 VERSION"

conda create --prefix ./venv python==3.9 -y

echo [$(date)]: "ACTIVATING THE ENVIRONMENT"

conda activate ./venv

echo [$(date)]: "INSTALLING LIBRARIES FROM requirement.txt"

pip install -r requirement.txt

echo [$(date)]: "END"