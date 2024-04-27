#!/bin/bash

huggingface-cli download --resume-download --local-dir-use-symlinks False GanymedeNil/text2vec-large-chinese --local-dir models/GanymedeNil/text2vec-large-chinese
huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm-xcomposer2-vl-7b-4bit --local-dir models/internlm/internlm-xcomposer2-vl-7b-4bit
