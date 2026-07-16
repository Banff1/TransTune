#!/usr/bin/env bash

set -e

# DATASETS=${DATASETS:-"*"}

SERVER_HOST=${SERVER_HOST:-"localhost"}

# SERVER_USERNAME=${SERVER_USERNAME:-"qdrant"}

SOURCE_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
DEFAULT_DOCKER_VOLUME_PARENT="/path/to/docker"

# Detect Python interpreter - prefer venv if available
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
elif [ -f "$SOURCE_DIR/../../venv/bin/python3" ]; then
    PYTHON_CMD="$SOURCE_DIR/../../venv/bin/python3"
elif [ -f "$SOURCE_DIR/../venv/bin/python3" ]; then
    PYTHON_CMD="$SOURCE_DIR/../venv/bin/python3"
elif [ -f "$SOURCE_DIR/venv/bin/python3" ]; then
    PYTHON_CMD="$SOURCE_DIR/venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

function run_exp() {
    # sync 
    # sudo bash -c "echo 1 > /proc/sys/vm/drop_caches" 
    export DOCKER_VOLUME_DIRECTORY="${DOCKER_VOLUME_DIRECTORY:-$DEFAULT_DOCKER_VOLUME_PARENT}"
    # Stop containers first to avoid "Device or resource busy" errors
    cd $SOURCE_DIR/engine/servers/milvus-single-node 2>/dev/null && docker-compose down > /dev/null 2>&1 || true
    # sudo rm -rf $SOURCE_DIR/results/* 2>/dev/null || true
    sudo rm -rf "${DOCKER_VOLUME_DIRECTORY}/volumes" 2>/dev/null || true
    sudo rm -rf $SOURCE_DIR/engine/servers/milvus-single-node/volumes 2>/dev/null || true

    SERVER_PATH=$1
    ENGINE_NAME=$2
    DATASETS=$3
    MONITOR_PATH=$(echo "$ENGINE_NAME" | sed -e 's/[^A-Za-z0-9._-]/_/g')
    nohup bash -c "cd $SOURCE_DIR/monitoring && rm -f docker.stats.jsonl && bash monitor_docker.sh" > /dev/null 2>&1 &
    cd $SOURCE_DIR/engine/servers/$SERVER_PATH ; docker-compose down > /dev/null; docker-compose up -d > /dev/null
    sleep 30
    $PYTHON_CMD -W ignore $SOURCE_DIR/run.py --engines "$ENGINE_NAME" --datasets "${DATASETS}" --host "$SERVER_HOST" > /dev/null
    # exit
    cd $SOURCE_DIR/engine/servers/$SERVER_PATH ; docker-compose down > /dev/null
    cd $SOURCE_DIR/monitoring && mkdir -p results && sudo mv docker.stats.jsonl ./results/${MONITOR_PATH}-docker.stats.jsonl
}

function get_result() {
    res_file=`ls $SOURCE_DIR/results/ | grep -v 'upload'` 
    cat $SOURCE_DIR/results/$res_file | grep -E "mean_precisions|rps|p95_time" | awk '{print $2}' | sed 's#,##g'
}


SERVER_PATH=${1:-milvus-single-node}
ENGINE_NAME=${2:-milvus-p10}
DATASETS=${3:-glove-25-angular}

run_exp $SERVER_PATH $ENGINE_NAME $DATASETS
get_result


# "nlist": 32768, "m":5, "nbits":8
# "nprobe": 16384