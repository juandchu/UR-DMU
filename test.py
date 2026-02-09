import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")


def test(net, test_loader, test_info, step, snippet_frame_count=16, model_file=None):
    # REMOVED 'wind' from arguments above
    with torch.no_grad():
        net.eval()
        net.flag = "Test"

        if model_file is not None and isinstance(model_file, str):
            net.load_state_dict(torch.load(model_file))

        frame_gt = np.load("frame_label/frame_gt.npy")
        frame_predict = []
        cls_label = []
        cls_pre = []

        for _data, _label, _name in test_loader:
            _data = _data.cuda()
            _label = _label.cuda()

            res = net(_data)
            a_predict = res["frame"].cpu().numpy()
            video_pred = a_predict.mean(0)

            cls_pre.append(1 if video_pred.max() > 0.5 else 0)

            fpre_ = np.repeat(video_pred, snippet_frame_count)
            frame_predict.extend(fpre_)

        frame_predict = np.array(frame_predict)

        if len(frame_predict) != len(frame_gt):
            x_old = np.linspace(0, 1, num=len(frame_predict))
            x_new = np.linspace(0, 1, num=len(frame_gt))
            f = interp1d(x_old, frame_predict, kind="linear")
            frame_predict = f(x_new)

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        if len(cls_label) > 0:
            accuracy = np.mean(np.array(cls_label) == np.array(cls_pre))
        else:
            accuracy = 0.0

        precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)

        # --- VISDOM REMOVED ---
        # wind.plot_lines("roc_auc", auc_score)
        # wind.plot_lines("accuracy", accuracy)
        # wind.plot_lines("pr_auc", ap_score)
        # wind.lines("scores", frame_predict)
        # wind.lines("roc_curve", tpr, fpr)

        # Terminal Feedback instead
        print(f"Step: {step} | AUC: {auc_score:.4f} | AP: {ap_score:.4f}")

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
