from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
from datasets.dataset.jde import letterbox
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
from getimagepath import getimage_path
from tracking_utils.utils import mkdir_if_missing
from opts import opts
from nms import non_max_suppression
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
###testApath
path_root="/home/ubuntu/deepsort/deep_sort_pytorch-master/deep_sort_pytorch-master/dataset/B-data/"
path_root_index=["Track2","Track3","Track6","Track8","Track11","Track12"]


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir='.', show_image=True, frame_rate=25):
    
    tracker = JDETracker(opt, frame_rate=frame_rate)
    p=path_root_index[5]
    if save_dir:
        save_dir = osp.join(save_dir, p)
        mkdir_if_missing(save_dir)
    image_path=getimage_path(path_root+p)
    timer = Timer()
    results = []
    frame_id = -1
    result_array_list=[]
    result=[]
    for path in image_path:

        # img=cv2.imread(path)

        img0 = cv2.imread(path)  # BGR
        # assert img0 is not None, 'Failed to load ' + img_path
        img_height=img0.shape[0]
        img_width=img0.shape[1]
        # print(img_height,img_width)
        # print(img0.shape)
        # Padded resize
        img, _, _, _ = letterbox(img0, height=608, width=1088)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0


        frame_id += 1

        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # if frame_id==2:
        #   break
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_cref=[]
        # result_array_list=[]
        for t in online_targets:
            
            tlwh = t.tlwh



            tid = t.track_id
            confidence = t.score

            vertical = tlwh[2] / tlwh[3] > 1.6
            if confidence<0.3:
              if tlwh[2] * tlwh[3] > 2700 and not vertical and tlwh[2] * tlwh[3]<100000:
                res=[frame_id,tid]
                res+=list(tlwh)
                res+=[1,0]
                online_tlwhs.append(tlwh)
                result_array_list.append(res)
                online_cref.append(confidence)
                # print(confidence)
                # result_array_list.append(tlwh)
                online_ids.append(tid)

            elif confidence>=0.3:
              if tlwh[2] * tlwh[3] > 1000 and not vertical and tlwh[2] * tlwh[3]<100000:
                  # print(tlwh[2] * tlwh[3])
                  res=[frame_id,tid]
                  res+=list(tlwh)
                  res+=[1,0]
                  online_tlwhs.append(tlwh)
                  result_array_list.append(res)
                  online_cref.append(confidence)
                  # print(confidence)
                  # result_array_list.append(tlwh)
                  online_ids.append(tid)
        # if frame_id==2:
        #   break
        timer.toc()
        # save results
        print(frame_id)
        # if result_array_list:

        online_tlwhs=np.array(online_tlwhs)
        online_ids=np.array(online_ids)
        online_cref=np.array(online_cref)
        # print(online_tlwhs)
        # print(online_tlwhs.shape)
        # print(online_ids.shape)
        # pick=non_max_suppression(online_tlwhs,0.7,online_cref)

        # online_tlwhsnms=online_tlwhs[pick]
        # online_idsnms=online_ids[pick]
        # online_crefnms=online_cref[pick]
        # result_array_list2=np.array(result_array_list).copy()[pick]
        # result+=list(result_array_list2)
        # print(result)

        # print(frame_id,online_idsnms)
        # result.append(online_tlwhsnms)
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, scores=online_cref,frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
            cv2.waitKey(1)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    res_array=np.array(result_array_list)
    # print(res_array)
      # print(res_array.shape)
    tmp_data1=np.where(res_array[:,[2]]<0,res_array[:,[2]]+res_array[:,[4]],res_array[:,[4]])
    res_array[:,[4]]=tmp_data1
    tmp_data=np.where(res_array[:,[3]]<0,res_array[:,[3]]+res_array[:,[5]],res_array[:,[5]])
    res_array[:,[5]]=tmp_data
    res_array[:,[2,3]]=np.maximum(res_array[:,[2,3]], 0)
    # print(res_array)
    res_array=np.round(res_array,0)
    # res_array=cutmorecord(res_array,img_width,img_height)
    # print(res_array)
    np.savetxt("{}.txt".format(p),res_array,fmt='%d,%d,%d,%d,%d,%d,%d,%d',delimiter=',')
  
    # save results
    # write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def cutmorecord(data,withd,height):
      tmp=data.copy()
      
      xywh=tmp[:,[2,3,4,5]]
      xyxy=np.zeros_like(xywh)
      # print()
      xyxy[:,[0]],xyxy[:,[1]],xyxy[:,[2]],xyxy[:,[3]]=xywh[:,[0]],xywh[:,[1]],xywh[:,[0]]+xywh[:,[2]],xywh[:,[1]]+xywh[:,[3]]
      
      xyxy[:,[2]]=np.minimum(xyxy[:,[2]],withd)
      xyxy[:,[3]]=np.minimum(xyxy[:,[3]],height)

      xyxy[:,[2]],xyxy[:,[3]]=xyxy[:,[2]]-xyxy[:,[0]],xyxy[:,[3]]-xyxy[:,[1]]

      tmp[:,[2,3,4,5]]=xyxy
      return tmp




def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT15_val_all_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
