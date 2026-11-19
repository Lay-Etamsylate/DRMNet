import numpy as np
import torch
import os
import traceback
import time
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from net.sanet import SANet
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from config import config
from net.model_config import VARIANTS, checkpoint_model_config
from utils.prediction_export import export_saved_predictions
import pandas as pd
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

this_module = sys.modules[__name__]
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--inference-only', action='store_true',
                    help='save predictions without running FROC; coordinates remain preprocessed voxels')
parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                    help='neural net')
parser.add_argument('--variant', choices=VARIANTS, default=None,
                    help='optional assertion; architecture is restored from checkpoint')
parser.add_argument('--test-anno', type=str, default=config['val_anno'][0],
                    help='annotation CSV used by the evaluation dataset reader')
parser.add_argument('--froc-annotations', default='/data/ltz-dg/filelist/annotations/annotations-luna16-lungseg.csv',
                    help='ground-truth CSV in the coordinate system used by the FROC evaluator')
parser.add_argument('--froc-excluded', default='/data/ltz-dg/filelist/annotations/luna16_excluded.csv')
parser.add_argument('--froc-seriesuids', default='/data/ltz-dg/filelist/annotations/seriesuids-luna16-lungseg.csv',
                    help='seriesuid CSV restricted to the evaluated split')
parser.add_argument("--mode", type=str, default = 'eval',
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default='./results/model/model.ckpt',
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test-set-name", type=str, default=config['test_set_name'],
                    help="path to save the results")
parser.add_argument("--fold-num", type=str, default='1',
                    help="fold-num")
parser.add_argument("--preprocessed-data-dir", type=str, default='1',
                    help="data path")


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    if args.mode == 'eval':
        data_dir = args.preprocessed_data_dir
        test_set_name = args.test_set_name
        num_workers = 16
        initial_checkpoint = args.weight
        net = args.net
        out_dir = args.out_dir

        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint, map_location='cpu')
            epoch = checkpoint['epoch']
            eval_cfg = checkpoint_model_config(config, checkpoint, args.variant)
            net = getattr(this_module, net)(eval_cfg).cuda()
            net.load_state_dict(checkpoint['state_dict'], strict=True)
            print('Model variant:', eval_cfg['model_variant'])
        else:
            print('No model weight file specified')
            return

        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir, 'res', str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'FROC')):
            os.makedirs(os.path.join(save_dir, 'FROC'))

        dataset = BboxReader(data_dir, test_set_name, args.test_anno, eval_cfg, 0, mode='eval')
#         test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
#                                  num_workers=num_workers, pin_memory=False, collate_fn=train_collate)
        eval(net, dataset, save_dir)
        if args.inference_only:
            print('Predictions saved to %s (preprocessed voxel coordinates; FROC skipped)' % save_dir)
        else:
            noduleCADEval(dataset, save_dir, epoch, out_dir, args.fold_num,
                          args.froc_annotations, args.froc_excluded, args.froc_seriesuids)
    else:
        logging.error('Mode %s is not supported' % (args.mode))


def eval(net, dataset, save_dir):
    net.set_mode('eval')
    net.use_rcnn = True

    manifest_path = os.path.join(save_dir, 'FROC', 'inference_manifest.json')
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    print('Total # of eval data %d' % (len(dataset)))
    for i, (input, truth_bboxes, truth_labels, crop_size) in enumerate(dataset):
        try:
            input = Variable(input).cuda().unsqueeze(0)
            truth_bboxes = np.array(truth_bboxes)
            truth_labels = np.array(truth_labels)
            pid = dataset.filenames[i]

            print('[%d] Predicting %s' % (i, pid))

            with torch.no_grad():
                net.forward(input, truth_bboxes, truth_labels)

            rpns = net.rpn_proposals.cpu().numpy()
            detections = net.detections.cpu().numpy()
            ensembles = net.ensemble_proposals.cpu().numpy()

#             print('detections', detections.shape)

            for kind, boxes in (('rpn', rpns[:, 1:]),
                                ('rcnn', detections[:, 1:-1]),
                                ('ensemble', ensembles[:, 1:])):
                boxes = boxes.copy()
                if len(boxes):
                    boxes[:, 1:4] += np.asarray(crop_size)
                else:
                    boxes = np.empty((0, 7), dtype=np.float32)
                # Save empty arrays too, so reruns cannot reuse stale detections.
                np.save(os.path.join(save_dir, '%s_%ss.npy' % (pid, kind)), boxes)

            # Clear gpu memory
            del input, truth_bboxes, truth_labels
            torch.cuda.empty_cache()

        except Exception as e:
            del input, truth_bboxes, truth_labels
            torch.cuda.empty_cache()
            traceback.print_exc()

            raise
    
    export_saved_predictions(save_dir, dataset.filenames)

def noduleCADEval(dataset, save_dir, epoch, out_dir, fold_num,
                  annotations_filename, annotation_exclude, val_path):
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
        os.makedirs(os.path.join(eval_dir, 'ensemble'))
        
    rpn_sens_boot,rpn_froc_boot,rpn_sens_norm,rpn_froc_norm = noduleCADEvaluation(annotations_filename,annotation_exclude,val_path, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    rcnn_sens_boot,rcnn_froc_boot,rcnn_sens_norm,rcnn_froc_norm = noduleCADEvaluation(annotations_filename,annotation_exclude,val_path, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    ensemble_sens_boot,ensemble_froc_boot,ensemble_sens_norm,ensemble_froc_norm = noduleCADEvaluation(annotations_filename,annotation_exclude,val_path, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))
    
#     sens_boot,froc_boot,sens_norm,froc_norm =noduleCADEvaluation(annotations_filename,'/root/workspace/imgctl/SANet/evaluationScript/annotations_excluded.csv', val_path, res_path, os.path.join(eval_dir, 'res'))
    
    content=[]
    content.append([str(epoch),'froc(avg)','0.125','0.25','0.5','1','2','4','8'])
    content.append(['rpn_boot']+[rpn_froc_boot]+rpn_sens_boot)
    content.append(['rcnn_boot']+[rcnn_froc_boot]+rcnn_sens_boot)
    content.append(['ensemble_boot']+[ensemble_froc_boot]+ensemble_sens_boot)
    content.append(['rpn_norm']+[rpn_froc_norm]+rpn_sens_norm)
    content.append(['rcnn_norm']+[rcnn_froc_norm]+rcnn_sens_norm)
    content.append(['ensemble_norm']+[ensemble_froc_norm]+ensemble_sens_norm)
    content_dataframe = pd.DataFrame(content)
    types = 'include_fp'
    if not os.path.exists(out_dir+'froc_'+types+'.csv'):
        content_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)
    else:
        exist_dataframe = pd.read_csv(out_dir+'froc_'+types+'.csv',header=None)
        new_dataframe = pd.concat([exist_dataframe,content_dataframe],axis=0)
        new_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)
        
    simple = []
    simple.append([str(epoch),rpn_froc_boot,rcnn_froc_boot,ensemble_froc_boot,rpn_froc_norm,rcnn_froc_norm,ensemble_froc_norm])
    simple_dataframe = pd.DataFrame(simple)
    types='simple'
    if not os.path.exists(out_dir+'froc_'+types+'.csv'):
        simple_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=['epoch','rpn_boot','rcnn_boot','ensemble_boot','rpn_norm','rcnn_norm','ensemble_norm'])
    else:
        exist_dataframe = pd.read_csv(out_dir+'froc_'+types+'.csv',header=None)
        new_dataframe = pd.concat([exist_dataframe,simple_dataframe],axis=0)
        new_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)

if __name__ == '__main__':
    main()
