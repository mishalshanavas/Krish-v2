#!/bin/bash

PROJECT_DIR="$HOME/Krish-v2"
VENV_PATH="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$PROJECT_DIR/app.py"


tmux has-session -t "krish" 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s "krish" "source $VENV_PATH/bin/activate && python3 $PYTHON_SCRIPT"
fi

