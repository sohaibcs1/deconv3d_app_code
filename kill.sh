#!/bin/bash

# Stop and delete the PM2 process
pm2 stop deconv3d
pm2 delete deconv3d

# Save PM2 state
pm2 save

echo "deconv3d stopped and removed from PM2."
