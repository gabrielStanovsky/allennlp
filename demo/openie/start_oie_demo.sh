#!/bin/bash
# Usage:
#       start_oie_demo.sh <path/to/model.tar.gz> <port>
# Run an Open IE model demo at the given port.

python -m allennlp.service.server_simple \
       --archive-path $1 \
       --predictor openie_predictor \
       --port $2 \
       --static-dir demo/openie/ \
