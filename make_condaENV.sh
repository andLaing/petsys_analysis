#!/usrbin/env bash

## Checks for miniconda and makes/activates crystal_cal environment if found.
## If not found where expected will give the user the option to either
## indicate a path to anaconda/miniconda or to install miniconda.

CONDA_ENV_NAME=crystal_cal
YML_FILENAME=environment-${CONDA_ENV_NAME}.yml

function activate_and_compile {
    conda activate ${CONDA_ENV_NAME}
    python setup.py develop
}

function make_conda_env {
    echo creating ${YML_FILENAME}

    cat <<EOF > ${YML_FILENAME}
name: ${CONDA_ENV_NAME}
dependencies:
- python       = 3.10
- scipy        = 1.10.0
- numpy        = 1.23.5
- matplotlib   = 3.7.1
- pyyaml       = 6.0
- pytest       = 7.1.2
- pandas       = 1.5.3
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
}

function activate_nonstandard_conda {
    echo "You requested to use a different (mini)conda path"
    echo "Please indicate the full path to the directory containing conda.sh:"
    read conda_path
    echo "Path requested: $conda_path, attempting source."
    if [ -d $conda_path ];
    then
        source $conda_path/conda.sh
    else
        echo "Path not found. Try again? [yes/no]"
        read retry1
        case $retry1 in
            *yes*) activate_nonstandard_conda;;
            *) kill -SIGINT $$;;
        esac
    fi
}

function install_miniconda {
    echo "Miniconda installation requested. Checking operating system."
    case "$(uname -s)" in
        *Darwin*) export CONDA_OS=MacOSX;;
        *Linux*) export CONDA_OS=Linux;;
        *) echo "Not prepared for your operating system, please install yourself"; kill -SIGINT $$;;
    esac
    CONDA_pf=$(uname -m)
    dwnl_path="https://repo.anaconda.com/miniconda/Miniconda3-latest-${CONDA_OS}-${CONDA_pf}.sh"
    if which wget >> /dev/null;
    then
        wget $dwnl_path -O miniconda.sh
    else
        curl $dwnl_path -o miniconda.sh
    fi
    inst_pth=$HOME/miniconda
    echo "Miniconda will be installed at $inst_pth, type path if want to change:"
    read new_path
    inst_pth=${new_path:-$inst_pth}
    bash miniconda.sh -b -p $inst_pth
    source $inst_pth/etc/profile.d/conda.sh
    echo "Miniconda installed, making environment"
    make_conda_env
    activate_and_compile
}

# Start main
if ! which conda >> /dev/null
then
    mini_loc=$HOME/miniconda
    if [ -d $mini_loc ];
    then
        CONDA_SH=$mini_loc/etc/profile.d/conda.sh
        source $CONDA_SH
    elif [ -d "$HOME/miniconda3" ];
    then
        CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
        source $CONDA_SH
    else
        echo "miniconda not found at $mini_loc or other known locations."
        echo "What do you want to do?"
        echo "Choose from: EXIT (stop script) or"
        echo "INSTALL (download and install miniconda) or"
        echo "USE (indicate path to conda.sh)"
        read do_me
        case $do_me in
            *EXIT*) return;;
            *INSTALL*) install_miniconda;;
            *USE*) activate_nonstandard_conda;;
            *) echo "Unrecognised request exiting"; return;;
        esac
    fi
fi

bmod_time=$(date -r make_condaENV.sh +%s)
if [ ! -f "$YML_FILENAME" ] || [ $bmod_time -gt $(date -r $YML_FILENAME +%s) ];
then
    echo "Environment file not found or older than make script."
    echo "Generating $YML_FILENAME and associated conda environment."
    make_conda_env
    conda deactivate
    activate_and_compile
else
    conda deactivate
    conda activate ${CONDA_ENV_NAME}
    util_time=$(date -r "$PWD/pet_code/src/io_util.pyx" +%s)
    c_time=$(date -r "$PWD/pet_code/src/io_util.c" +%s)
    if [ $util_time -gt $c_time ];
    then
        python setup.py develop
    fi
fi

echo "Environment installed and activated, running tests."
pytest -v
