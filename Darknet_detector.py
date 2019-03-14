# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import glob
import sys
#import re
from darknet import Darknet


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Darknet_detector")
    parser.add_argument("-I", "--inFrmDir",    help="入力フレームが存在するディレクトリ",required=True)
    parser.add_argument("-O", "--OutMOT",      help="出力MOTデータ",required=True)
    parser.add_argument("-c", "--config",      help="config ファイル",default="cfg/yolov3-voc.cfg")
    parser.add_argument("-m", "--model",       help="model ファイル",default="yolov3.weights")
    parser.add_argument("-t", "--thresh",      help="thresh",type=float,default=.5)
    parser.add_argument("-ht", "--hier_thresh", help="hier_thresh",type=float,default=.5)
    parser.add_argument("-n", "--nms",         help="nms",type=float,default=.45)
#    parser.add_argument("-F", "--FPS",   help="フレームレート",type=int, default=15)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("------------------------------")
#    print("in FrmDir     　:%s" % args.inFrmDir)
    print("out MOT data    :%s" % args.OutMOT)
    print("------------------------------")
    print("config          :%s" % args.config)
    print("model           :%s" % args.model)
    print("------------------------------")
    print("thresh          :%f" % args.thresh)
    print("hier_thresh     :%f" % args.hier_thresh)
    print("nms             :%f" % args.nms)
    print("------------------------------")
  
#    print("フレームレート　　　　　　　:%d" % args.FPS)
    
#入力フレームディレクトリ確認
    if (os.path.exists(args.inFrmDir) == False):
        print("入力ディレクトリが存在しません。: %s" % args.inFrmDir)
        sys.exit()
    else:
        FrameImages = sorted(glob.glob(os.path.join(args.inFrmDir, '*.jpg')))
        
    if(len(FrameImages) == 0):
        print( "入力フレームが存在しません。: %s" % args.inFrmDir)
        sys.exit()
    
    numFrames = len(FrameImages)
    print("numFrames:", numFrames)
     
#出力フレームディレクトリ確認　無い場合には作る
    OutDir = os.path.dirname(args.OutMOT)
 
    if (os.path.exists(OutDir) == False):
        os.makedirs(OutDir)
    OutMOT = open(args.OutMOT, 'w')  


    #Daraknet detectorのインスタンス
    darknet = Darknet(config=args.config, model=args.model,thresh=args.thresh,hier_thresh=args.hier_thresh,nms=args.nms)

    accum=0
    for i in range(numFrames):
        imgName = os.path.basename(FrameImages[i])
        fno = int(imgName.split(".")[0])
        st_time = cv2.getTickCount()
        r = darknet.detect(FrameImages[i])
        ed_time = cv2.getTickCount()
        
        diff = (ed_time-st_time)
        accum += diff
        
        numObj = len(r)
        for j in range(numObj):
            TLx = r[j][2][0] - (r[j][2][2]/2)
            TLy = r[j][2][1] - (r[j][2][3]/2)
            OutMOT.write("%d,-1,%5.2f,%5.2f,%5.2f,%5.2f,1,-1,-1\n" %(fno, TLx,TLy,r[j][2][2],r[j][2][3]))
#             file.write(string)
            print("fno:%d nObj:%d time:%f ms" % (i,j, diff * 1000.0 / cv2.getTickFrequency() ))
#        FrameImage = cv2.imread(FrameImages[i])
    OutMOT.close()
    AveTime = (accum/numFrames) * 1000.0 / cv2.getTickFrequency() 
    print("ave  time:%f ms" % AveTime)
 
#     r = darknet.detect("data/000280.jpg")
#     print r
#     r = darknet.detect("data/000226.jpg")
#     print r
