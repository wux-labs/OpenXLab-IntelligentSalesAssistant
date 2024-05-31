#!/bin/bash

huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm/internlm2-chat-20b --local-dir models/internlm/internlm2-chat-20b
huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm2-chat-7b --local-dir models/internlm/internlm2-chat-7b

huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm-xcomposer2-vl-7b-4bit --local-dir models/internlm/internlm-xcomposer2-vl-7b-4bit
huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm-xcomposer2-vl-7b --local-dir models/internlm/internlm-xcomposer2-vl-7b

huggingface-cli download --resume-download --local-dir-use-symlinks False stabilityai/stable-diffusion-2-1 --local-dir models/stabilityai/stable-diffusion-2-1

huggingface-cli download --resume-download --local-dir-use-symlinks False GanymedeNil/text2vec-large-chinese --local-dir models/GanymedeNil/text2vec-large-chinese
