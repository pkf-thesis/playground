#!/bin/bash
#perl -pi.bak -e 's/\r$//' screenstart.sh
#if ! ps -ef | grep -v grep | grep $1 ; then
#   cd /usr/local/share/FKP/src
    echo Adding $1 to screens...
    screen -dmS $1 python main.py -d=gtzan -logging=../logs
    screen -list
#fi
exit 0