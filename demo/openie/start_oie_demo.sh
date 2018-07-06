#!/bin/bash
# Usage:
#  start_oie_demo.sh path/to/model.tar.gz
# Run an open ie model demo

python -m allennlp.service.server_simple \
       --archive-path $1 \
       --predictor openie_predictor \
       --port 8008 \
       --static-dir demo/openie/ \
