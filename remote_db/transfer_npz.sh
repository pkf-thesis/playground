#!/bin/bash
#argument 1: INITIALS
#argument 2: DATA
#example: bash transfer_npz.sh frda gtzan
scp -r npys/$2 $1:/usr/local/share/FKP/npzs