#!/bin/bash
set -e

bash /apollo/scripts/bootstrap.sh stop
bash /apollo/scripts/bootstrap.sh

cyber_recorder play -f docs/demo_guide/test0.record -l
