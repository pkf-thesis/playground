#!/bin/bash
screen -list
echo Removing $1 from screens...
screen -S $1 -X quit
screen -list