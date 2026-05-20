#!/bin/bash

# Exit immediately if something fails
set -e

# Activate virtual environment
source deconv3d/bin/activate

mkdir -p logs

# 🔒 Force app to use ONLY GPU 7
export CUDA_VISIBLE_DEVICES=7

# Start app with PM2
pm2 start app.py \
  --name deconv3d \
  --interpreter python \
  --output logs/logs.txt \
  --error logs/error.txt \
  --time

# Save PM2 process list
pm2 save
