# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
#import pickle
import os
import cv2
#from os import getcwd
#from os.path import join
import argparse
import sys

sets=[]
labelsDir=[]
AnnotationsDir=[]
JPEGImageDir=[]
classes = []


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    global classes
#    in_file = open('VOCdevkit/Annotations/%s.xml'%(image_id))
#    out_file = open('VOCdevkit/labels/%s.txt'%(image_id), 'w')
    in_file = open('%s/%s.xml'%(AnnotationsDir,image_id))
    out_file = open('%s/%s.txt'%(labelsDir,image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
#    size = root.find('size')
#    w = int(size.find('width').text)
#    h = int(size.find('height').text)

    #annotationに画像サイズが入っていなかった時の措置。実際の画像からサイズを得る。
#    if(w==0 or h==0):
    #画像サイズは実際の画像から取得
    print(image_id)
    bgr_image = cv2.imread(('%s/%s.jpg'%(JPEGImageDir,image_id)), cv2.IMREAD_COLOR)
    h = bgr_image.shape[0]   
    w = bgr_image.shape[1]

    for obj in root.iter('object'):
        # difficultTagの'difficult'でも'occluded'でもOKとする。
        difficultTag = obj.find('difficult')
        if(difficultTag == None):
            difficultTag = obj.find('occluded')
            if(difficultTag == None):
                sys.exit()
        difficult = difficultTag.text    
            
        if int(difficult) == 1:
            continue
            
        cls = obj.find('name').text
        if cls not in classes: 
            classes.append(cls)

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOT_rescaler")
    parser.add_argument("-I", "--RootDir", help="VOC ROOTディレクトリ",required=True )
    parser.add_argument("-O", "--OUTDir", help="モデル出力ディレクトリ",default="backup" )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("RootDir  :%s" % args.RootDir)
    
    AnnotationsDir = args.RootDir + "/Annotations"
    if not os.path.exists(AnnotationsDir):
        print("-- Annotations dir not found --:%s" % AnnotationsDir)
        sys.exit()
        
    ImageSets = args.RootDir + "/ImageSets"
    if not os.path.exists(ImageSets):
        print("-- ImageSets dir not found --:%s" % ImageSets) 
        sys.exit()
    else:
        ImageSetsMain = ImageSets +"/Main"
        if not os.path.exists(ImageSetsMain):
            print("-- ImageSet内にMain dir が無い --:%s" % ImageSetsMain) 
            sys.exit()
        else:
            for x in os.listdir(ImageSetsMain):  
                setfile = os.path.join(ImageSetsMain,x)
                if os.path.isfile(setfile):
                    filename, ext = os.path.splitext(x)
                    if ext == ".txt":
                        sets.append(filename)
            print(sets)

    JPEGImageDir = args.RootDir + "/JPEGImages"
    if not os.path.exists(JPEGImageDir):
        print("-- JPEGImages dir not found --:%s" % JPEGImageDir) 
        sys.exit()
    
    labelsDir = args.RootDir + "/labels"
    if not os.path.exists(labelsDir):
        os.makedirs(labelsDir)

    for image_set in sets:
            image_ids = open('%s/%s.txt'%(ImageSetsMain,image_set)).read().strip().split()
            list_file = open('%s/%s.txt'%(args.RootDir,image_set), 'w')
            for image_id in image_ids:
                list_file.write('%s/%s.jpg\n'%(JPEGImageDir, image_id))
                convert_annotation(image_id)
            list_file.close()

    #voc.data作成
    vocdata_file = open('%s/voc.data'%(args.RootDir), 'w')
    vocdata_file.write("classes= %d\n" % len(classes))
    vocdata_file.write("train  = %s/train.txt\n" % (args.RootDir))
    vocdata_file.write("valid  = %s/val.txt\n" % (args.RootDir))
    vocdata_file.write("names = %s/voc.names\n" % (args.RootDir))
    vocdata_file.write("backup = %s\n" % (args.OUTDir))
    vocdata_file.close()
    #voc.name作成
    vocname_file = open('%s/voc.names'%(args.RootDir), 'w')
    for x in classes:
        vocname_file.write(str(x) + "\n")
    vocname_file.close()
    
    if not os.path.exists(args.OUTDir):
        os.makedirs(args.OUTDir)
