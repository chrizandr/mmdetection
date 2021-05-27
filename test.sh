#!/bin/bash
#SBATCH -A research
#SBATCH --nodelist=gnode55
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -n 8
#SBATCH --time=2-00:00:00

module load gflags/2.2.1 &&

module load glog/0.3.5 &&

module load cuda/10.2 && module load cudnn/7.6.5-cuda-10.2 &&

module load openmpi/4.0.0 &&

module load python/3.7.4


source ~/mmd/bin/activate


echo "Copying data files"
if [ -d "/ssd_scratch/cvit/chrizandr" ]
then
  echo "Clearing existing files on node";
  rm -r /ssd_scratch/cvit/chrizandr
fi

images="england_vs_croatia"
prefix="envscr"

mkdir -p /ssd_scratch/cvit/chrizandr/
rsync -az chrizandr@ada:/share3/chrizandr/sports/dataset/$images/images/ /ssd_scratch/cvit/chrizandr/images

echo "Done copying data files"


# python infer.py --cfg configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py --weights checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --annot_file `echo $prefix`_frcnn.xml
# python infer.py --cfg configs/retinanet/retinanet_r50_fpn_1x_coco.py --weights  checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --annot_file `echo $prefix`_retina.xml
python infer.py --cfg soccerdb_config_retina.py --weights checkpoints/retinanet_x101_64x4d_fpn_1x.pth --annot_file `echo $prefix`_soccerdb.xml

# frvscr 54
# envscr 55
