import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np

VOC_CLASSES = (
    "beasts",
    "bird",
    "fish",
    "insect",
    "plant",
    "person",
)


def eval():
    # self._write_voc_results_file(all_boxes)
    root = '/opt/tiger/minist/datasets/groot_voc'
    # res_path = '/opt/tiger/minist/datasets/groot_voc/results'
    res_path = '/opt/tiger/minist/datasets/results/result_yoloxl_0.001_nms_0.45/'
    # res_path = '/opt/tiger/minist/datasets/results/results_yolox-_official_0.001_0.65'
    IouTh = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    mAPs = []
    for iou in IouTh:
        mAP = do_python_eval(root, res_path, iou)
        mAPs.append(mAP)

    print("--------------------------------------------------------------")
    print("map_5095:", np.mean(mAPs))
    print("map_50:", mAPs[0])
    print("--------------------------------------------------------------")
    return np.mean(mAPs), mAPs[0]


def get_voc_results_file_template(res_path):
    filename = "comp4_det_test" + "_{:s}.txt"
    filedir = res_path
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

def do_python_eval(root, res_path, iou=0.5):
    
    rootpath = root
    name = 'val'
    annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
    imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
    cachedir = os.path.join(
        root, "annotations_cache"
    )
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = True if int(self._year) < 2010 else False
    use_07_metric = False
    print("Eval IoU : {:.2f}".format(iou))
    # if output_dir is not None and not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)
    for i, cls in enumerate(VOC_CLASSES):

        if cls == "__background__":
            continue

        filename = get_voc_results_file_template(res_path).format(cls)
        rec, prec, ap = voc_eval(
            filename,
            annopath,
            imagesetfile,
            cls,
            cachedir,
            ovthresh=iou,
            use_07_metric=use_07_metric,
        )
        aps += [ap]
        if iou == 0.5:
            print("AP for {} = {:.4f}".format(cls, ap))
        # if output_dir is not None:
        #     with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
        #         pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
    if iou == 0.5:
        print("Mean AP = {:.4f}".format(np.mean(aps)))
        print("~~~~~~~~")
        print("Results:")
        for ap in aps:
            print("{:.3f}".format(ap))
        print("{:.3f}".format(np.mean(aps)))
        print("~~~~~~~~")
        print("")
        print("--------------------------------------------------------------")
        print("Results computed with the **unofficial** Python eval code.")
        print("Results should be very close to the official MATLAB eval code.")
        print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
        print("-- Thanks, The Management")
        print("--------------------------------------------------------------")

    return np.mean(aps)


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    cachedir,
    ovthresh=0.5,
    use_07_metric=False,
):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print("Reading annotation for {:d}/{:d}".format(i + 1, len(imagenames)))
        # save
        print("Saving cached annotations to {:s}".format(cachefile))
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

        # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects

if __name__ == "__main__":
    eval()