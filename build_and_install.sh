cd pointnet2_lib/pointnet2
python setup.py install
cd ../../

cd lib/utils/iou3d/
python setup.py install

cd ../roipool3d/
python setup.py install

cd ../../../tools

python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt PointRCNN.pth --batch_size 1 --eval_mode rcnn --set RPN.LOC_XZ_FINE False

python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rpn/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rcnn 
