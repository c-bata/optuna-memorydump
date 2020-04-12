#!/bin/sh

DIR=$(cd $(dirname $0); pwd)
REPOSITORY_ROOT=$(cd $(dirname $(dirname $0)); pwd)

set +e

rm db.sqlite3

if [ ! -d ${REPOSITORY_ROOT}/venv ]; then
    echo "Create virtualenv"
    python3.7 -m venv venv
    source venv/bin/activate
    python setup.py develop
    pip install bokeh
else
    echo "Activate virtualenv"
    source venv/bin/activate
fi

set -ex

python ${DIR}/rosenbrock.py

optuna dashboard --storage sqlite:///db.sqlite3 --study dumped