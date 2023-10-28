# Data Science end to end Project

## Project Componsents/Pipeline (Hierarchy of a complete project):
* Data Ingestion
* EDA
* Feature Engineering
* Model Selection
* Model Building
* Evaluation

### Necessary files for project
* Logger file
* Exception file
* util 
* setup.py
* requirement.txt

## Project Structure
* .github
    * workflow
        * yaml files (.gitkeep)
* Notebook
    * research.ipynb
* src
    * project_name folder
        * components
            * data_ingestion.py
            * data_preprocessing.py
            * model_training.py
        * pipeline
            * training.py
            * prediction.py
        * exception.py
        * logger.py
        * utils.py
* requiremet.txt
* setup.py
* LICANCE
* README.md


# Instruction to initialize the git

## Initialize Git Repo in VSCode
```
git init
```

## Add any file to stagin area
```
git add abc.txt
git add .
```
## Commit the changes that you want to push to Git
```
git commit -m "this is my first commit"
```

## Push changes to Git
```
git push
```

## Pull all the latest changes from Git Repo
```
git pull
```

# Configure and Setup your Project

## run .sh file that will automatically initialize your virtual environment
```
bash your_file_name.sh
```

# Run the python template that will create all the required files and folder required for Industry Grade Project
## Its a standard templat used for any porject
```
python template.py
```

# install your local package onto virtual environment

## Method 1
```
python setup.py install
```

## Method 2: add "-e ." to you requirement.txt file and run below command.

```
pip install -r requirements.txt
```