#!/bin/bash
ffmpeg -i image_at_epoch_%04d.png -vf format=yuv420p -q:v 0 -r 10 animation.mp4
