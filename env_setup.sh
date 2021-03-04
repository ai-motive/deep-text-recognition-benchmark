#!/bin/bash

ROOT_DIR=$(pwd)
export PYTHONPATH=${ROOT_DIR}
echo

CRNN_LEARN_DIR=${ROOT_DIR}/crnn_learn
export PYTHONPATH=${PYTHONPATH}:${CRNN_LEARN_DIR}
echo $PYTHONPATH

