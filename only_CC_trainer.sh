#!/bin/bash
#alg514
# python batch_trainer.py --set ./dataset/alg514 --config ./config/alg514/base.json --model ept expr vanilla
#CC
python batch_trainer.py --set ./dataset/CC --config ./config/CC/CC/base.json --model ept expr vanilla
#IL
# python single_trainer.py --set ./dataset/IL --config ./config/IL/IL/base.json --model ept expr vanilla
