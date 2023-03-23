#!/usrbin/env bash

# Make a conda environment for the convertor assuming miniconda installed
# in home folder.
if ! which conda >> /dev/null
then
    if [ -d "$HOME/miniconda" ];
    then
        CONDA_SH=$HOME/miniconda/etc/profile.d/conda.sh
        source $CONDA_SH
    else
        echo "miniconda not found in home. Please install."
        ## Will sort the instalation but not yet.
    fi
fi

CONDA_ENV_NAME=crystal_cal
YML_FILENAME=environment-${CONDA_ENV_NAME}.yml

echo creating ${YML_FILENAME}

cat <<EOF > ${YML_FILENAME}
name: ${CONDA_ENV_NAME}
dependencies:
- python       = 3.8
- scipy        = 1.7.1
- numpy        = 1.21.2
- matplotlib   = 3.5
- pyyaml       = 6.0
- natsort      = 7.1
- pytest       = 6.2.5
- pandas       = 1.3.5
- docopt       = 0.6.2
- configparser = 5.0.2
- cython       = 0.29.32
- pyarrow      = 8.0.0
EOF

if conda env list | grep ${CONDA_ENV_NAME};
then
    conda env update --name ${CONDA_ENV_NAME} --file ${YML_FILENAME} --prune
else
    conda env create -f ${YML_FILENAME}
fi
conda activate ${CONDA_ENV_NAME}

# This is only really necessary
# if there are changes to the
# package structure.
python setup.py develop
