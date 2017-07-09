import os
import numpy as np
import xml.etree.ElementTree as et


def ind_to_class(ind=None):
    i_to_c = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor']
    if ind is None:
        return i_to_c
    else:
        return i_to_c[ind]


def class_to_ind(class_name=None):
    c_to_i = {c:i for i,c in enumerate(ind_to_class())}
    if class_name is None:
        return c_to_i
    else:
        return c_to_i[class_name]


def train_2007(devkit_path, include_difficult=True):
    dataset = _reader(devkit_path, 2007, 'train')
    if not include_difficult:
        dataset = _discard_difficult(dataset)
    return dataset


def valid_2007(devkit_path, include_difficult=True):
    dataset = _reader(devkit_path, 2007, 'val')
    if not include_difficult:
        dataset = _discard_difficult(dataset)
    return dataset


def trainval_2007(devkit_path, include_difficult=True):
    dataset = _reader(devkit_path, 2007, 'trainval')
    if not include_difficult:
        dataset = _discard_difficult(dataset)
    return dataset


def test_2007(devkit_path, include_difficult=True):
    dataset = _reader(devkit_path, 2007, 'test')
    if not include_difficult:
        dataset = _discard_difficult(dataset)
    return dataset


def train_2012(devkit_path, include_difficult=True):
    dataset = _reader(devkit_path, 2012, 'train')
    if not include_difficult:
        dataset = _discard_difficult(dataset)
    return dataset


def valid_2012(devkit_path, include_difficult=True):
    dataset = _reader(devkit_path, 2012, 'val')
    if not include_difficult:
        dataset = _discard_difficult(dataset)
    return dataset


def trainval_2012(devkit_path, include_difficult=True):
    dataset = _reader(devkit_path, 2012, 'trainval')
    if not include_difficult:
        dataset = _discard_difficult(dataset)
    return dataset


def test_2012(devkit_path, include_difficult=True):
    dataset = _reader(devkit_path, 2012, 'test')
    if not include_difficult:
        dataset = _discard_difficult(dataset)
    return dataset


def eval_detection(dataset, results, num_classes=20, overlap_thres=0.5, use_difficult=False, use_07_metric=False, ignore_id=False):
    """ Vectorized code for PASCAL VOC detection evaluation """
    ## Input
    ##  dataset: list of data
    ##  data: dict
    ##      id: str = image file name
    ##      object: list of object
    ##          bbox: [x1,y1,x2,y2]
    ##          label: int
    ##          difficult: bool
    ##  results: list of result
    ##  result: dict
    ##      id: str = image file name
    ##      object: list of object
    ##          bbox: [x1,y1,x2,y2]
    ##          label: int
    ##          conf: float
    ## Output
    ##  evaluation: list of dict (length = num_classes)
    ##      'recall', 'precision', 'AP'

    ## check both samples are equal
    if not ignore_id:
        assert np.all([s['id'] == r['id'] for s,r in zip(dataset, results)]),'dataset and result ID must be matched'

    ## start matching
    all_true = [np.array([], np.bool) for _ in xrange(num_classes)]
    all_conf = [np.array([], np.float) for _ in xrange(num_classes)]
    for data, result in zip(dataset,results):
        all_gt_objects = np.array([[obj['label']]+obj['bbox']+[obj['difficult']] for obj in data['objects']], np.float)
        all_pred_objects = np.array([[obj['label']]+obj['bbox']+[obj['conf']] for obj in result['objects']], np.float)
        if len(all_gt_objects) and len(all_pred_objects):
            for label in xrange(num_classes):
                gt_objects = all_gt_objects[all_gt_objects[:,0]==label]
                pred_objects = all_pred_objects[all_pred_objects[:,0]==label]
                num_gt = len(gt_objects) ## N
                num_pred = len(pred_objects) ## M
                if num_gt and num_pred:
                    if not use_difficult:
                        ## number of difficult gt
                        num_difficult = np.count_nonzero(gt_objects[::,-1])

                        ## sort gt objects by difficult
                        if num_difficult:
                            inds = gt_objects[::,-1].argsort()[::-1]
                            gt_objects = gt_objects[inds]

                    ## sort pred objects by confidence
                    inds = pred_objects[::,-1].argsort()[::-1]
                    pred_objects = pred_objects[inds]

                    ## compute overlap
                    overlaps = _jaccard_overlap(pred_objects[:,1:5], gt_objects[:,1:5]) # (N,M)

                    ## gt inds for max overlap
                    gt_inds = overlaps.argmax(axis=1) # (N,)

                    ## threshold overlap
                    gt_overlaps = overlaps[np.arange(len(pred_objects)), gt_inds] # (N,)

                    ## overlaps below the thres are false
                    mask = gt_overlaps < overlap_thres
                    false_inds = np.nonzero(mask)[0].tolist()
                    thres_inds = np.nonzero(~mask)[0]

                    ## discard pred objects matched with difficult gt
                    if not use_difficult and num_difficult:
                        thres_inds = thres_inds[gt_inds[thres_inds] >= num_difficult]

                    ## max overlap
                    valid_gt_inds = np.unique(gt_inds[thres_inds])
                    matched_gt_inds = list()
                    true_inds = list()
                    for gt_ind, pred_ind in zip(gt_inds[thres_inds], thres_inds):
                        if len(matched_gt_inds) == len(valid_gt_inds):
                            false_inds.append(pred_ind)
                        else:
                            if gt_ind not in matched_gt_inds:
                                matched_gt_inds.append(gt_ind)
                                true_inds.append(pred_ind)
                            else:
                                false_inds.append(pred_ind)

                    ## store matching results
                    if len(true_inds):
                        all_true[label] = np.append(all_true[label], [True]*len(true_inds))
                        all_conf[label] = np.append(all_conf[label], pred_objects[true_inds][:,-1])
                    if len(false_inds):
                        all_true[label] = np.append(all_true[label], [False]*len(false_inds))
                        all_conf[label] = np.append(all_conf[label], pred_objects[false_inds][:,-1])
                elif num_pred:
                    all_true[label] = np.append(all_true[label], [False]*num_pred)
                    all_conf[label] = np.append(all_conf[label], pred_objects[:,-1])

    ## labels of all ground truth objects
    all_obj_label = np.array([obj['label'] for data in dataset for obj in data['objects']])
    if not use_difficult:
        all_obj_diff = np.array([obj.get('difficult', False) for data in dataset for obj in data['objects']])
        all_obj_label = all_obj_label[~all_obj_diff]

    ## evaluation
    evals = list()
    for label in xrange(num_classes):
        num_gt_objects = np.count_nonzero(all_obj_label == label)

        true = all_true[label]
        conf = all_conf[label]

        if num_gt_objects and len(true):
            true = true[conf.argsort()[::-1]]

            tp = np.cumsum(true)
            fp = np.cumsum(~true)
            rec = tp/float(num_gt_objects)
            prec = tp/np.maximum(tp+fp, np.finfo(np.float64).eps)
            ap = _VOCap(rec, prec, use_07_metric)
        else:
            rec = np.array([])
            prec = np.array([])
            ap = 0

        evals.append({
            'recall': rec[::-1].tolist(),
            'precision': prec[::-1].tolist(),
            'AP': ap,
        })

    return evals


def _reader(devkit_path, year, split):
    ## Input
    ##  year: integer or string = 2007, 2012
    ##  split: string = 'train', 'val', 'trainval', 'test'

    ## target VOC path
    voc_path = os.path.join(devkit_path, 'VOC'+str(year))

    ## class information path
    info_path = os.path.join(voc_path, 'ImageSets', 'Main')

    ## image path
    img_path = os.path.join(voc_path, 'JPEGImages')

    ## annotation path
    anno_path = os.path.join(voc_path, 'Annotations')

    ## load information
    sample_info = np.loadtxt(os.path.join(info_path, split+'.txt'),
                            dtype=np.object)
    num_image = sample_info.shape[0]

    ## class to label mapper
    c_to_i = class_to_ind()

    ## list for store whole sample
    samples = list()

    ## for each image, convert to sample
    for k in range(num_image):
        ## dictionary for store sample information
        sample = dict()

        ## set id
        name = sample_info[k]
        sample['id'] = name

        ## load annotation
        anno_file_path = os.path.join(anno_path, name+'.xml')
        if os.path.isfile(anno_file_path):
            tree = et.parse(anno_file_path)
            root = tree.getroot()

            ## image info
            image_file_name = root.findtext('filename')
            sample['image'] = os.path.join(img_path, image_file_name)
            size_info = root.find('size')
            sample['width'] = int(size_info.findtext('width'))
            sample['height'] = int(size_info.findtext('height'))

            ## get object information
            obj_list = root.findall('object')

            if len(obj_list):
                ## list for store objects
                sample['objects'] = list()

                ## for each object in image, get label and bbox
                for obj_info in obj_list:
                    ## dictionary for storing object information
                    obj = dict()

                    ## get object name
                    obj_class = obj_info.findtext('name')
                    obj['class'] = obj_class

                    ## get label from object name and store
                    obj['label'] = c_to_i[obj_class]

                    ## get object bbox and store(-1 for 0 based index)
                    bbox = obj_info.find('bndbox')
                    x1 = int(bbox.findtext('xmin'))-1
                    y1 = int(bbox.findtext('ymin'))-1
                    x2 = int(bbox.findtext('xmax'))-1
                    y2 = int(bbox.findtext('ymax'))-1
                    obj['bbox'] = [x1, y1, x2, y2]

                    ## get difficulty and trancated
                    obj['difficult'] = bool(int(obj_info.findtext('difficult')))
                    obj['truncated'] = bool(int(obj_info.findtext('truncated')))

                    ## store object information
                    sample['objects'].append(obj)

        ## store sample
        samples.append(sample)

    return samples


def _discard_difficult(dataset):
    for data in dataset:
        if data.has_key('objects'):
            filtered_objects = [obj for obj in data['objects'] if obj['difficult'] == False]
            data['objects'] = filtered_objects
    return dataset


def _jaccard_overlap(boxes1, boxes2):
    """
    compute jaccard overlap(IOU)
    """
    ## Input
    ##  boxes1: 1d or 2d array = n*(x1,y1,x2,y2)
    ##  boxes2: 1d or 2d array = n*(x1,y1,x2,y2)
    do_squeeze = False
    if boxes1.ndim == 1:
        boxes1 = boxes1[np.newaxis]
        do_squeeze = True
    if boxes2.ndim == 1:
        boxes2 = boxes2[np.newaxis]
        do_squeeze = True

    boxes1_w = np.maximum(0., boxes1[:, 2] - boxes1[:, 0] + 1)
    boxes1_h = np.maximum(0., boxes1[:, 3] - boxes1[:, 1] + 1)
    boxes1_area = boxes1_w * boxes1_h

    boxes2_w = np.maximum(0., boxes2[:, 2] - boxes2[:, 0] + 1)
    boxes2_h = np.maximum(0., boxes2[:, 3] - boxes2[:, 1] + 1)
    boxes2_area = boxes2_w * boxes2_h

    inter_tl = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    inter_br = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_w = np.maximum(0., inter_br[:, :, 0] - inter_tl[:, :, 0] + 1)
    inter_h = np.maximum(0., inter_br[:, :, 1] - inter_tl[:, :, 1] + 1)
    inter_area = inter_w * inter_h

    iou = inter_area / (boxes1_area[:, None] + boxes2_area[None, :] - inter_area)
    if do_squeeze:
        iou = np.squeeze(iou)
    return iou


def _VOCap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec.ravel(), [1.]))
        mpre = np.concatenate(([0.], prec.ravel(), [0.]))
        for i in range(1,len(mpre)):
            mpre[-i-1] = max(mpre[-i-1],mpre[-i])
        idx = [i for i in range(1,len(mrec)) if mrec[i-1] != mrec[i]]
        ap = sum([(mrec[i]-mrec[i-1])*mpre[i] for i in idx])

    return ap