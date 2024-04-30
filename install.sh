#!/bin/bash
python_version="3.9"
# Initialize conda
eval "$(conda shell.bash hook)"
#print the current environment
echo "The current environment is $CONDA_DEFAULT_ENV."
while [[ "$CONDA_DEFAULT_ENV" != "base" ]]; do
    conda deactivate
done

#if conde transforna not found in the list of environments, then create the environment
if [[ $(conda env list | grep "transforna") == "" ]]; then
    conda create -n transforna python=$python_version -y
    conda activate transforna
    conda install -c anaconda setuptools -y


    
fi
conda activate transforna

echo "The current environment is transforna."
pip install setuptools==59.5.0
# Uninstall TransfoRNA using pip
pip uninstall -y TransfoRNA

rm -rf dist TransfoRNA.egg-info


# Reinstall TransfoRNA using pip
python setup.py sdist
pip install dist/TransfoRNA-0.0.1.tar.gz