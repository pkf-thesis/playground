#!/bin/bash
scp -r -oProxyJump=$1@130.226.142.166 src $1@10.1.1.121:/usr/local/share/FKP

ssh -J $1@130.226.142.166 $1@10.1.1.121  "cd /usr/local/share/FKP && chgrp -R FKP src && chmod -R g+w src"
#chmod g+w src
