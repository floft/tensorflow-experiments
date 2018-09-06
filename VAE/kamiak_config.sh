#
# config file
#
dir="/data/vcea/matt.taylor/Projects/ras-object-detection/vae"
program="VAE.py"
modelFolder="vae-model2"
logFolder="vae-log2"
compressedDataset="dataset.zip"
dataset=(*.gz)

# Connecting to the remote server
# Note: this is rsync, so make sure all paths have a trailing slash
remotedir="$dir/"
remotessh="kamiak"
localdir="/home/garrett/Documents/School/18_Fall/Research/tensorflow-experiments/VAE/"
