import pdb
import numpy as np
import torch.utils.data as data
import components.utils as utils
from components.options import *
from components.config import *
from components.train import *
from components.test_function import test
from components.model import *
from components.utils import Visualizer
import os
from components.dataset_loader import *
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    # print(config)
    worker_init_fn = None
    gpus = [0]
    torch.cuda.set_device("cuda:{}".format(gpus[0]))
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    # Model creation
    config.len_feature = 1024
    net = WSAD(
        config.len_feature, flag="Train", a_nums=config.a_nums, n_nums=config.n_nums
    )  # 60 = memory prototypes
    net = net.cuda()

    normal_train_loader = data.DataLoader(
        FeatureDataset(
            data_dir=config.root_dir,
            mode="Train",
            modal=config.modal,
            num_segments=config.num_segments,
            len_feature=config.len_feature,
            is_normal=True,
        ),
        batch_size=4,  # original was 64
        shuffle=True,  # just shuffles the videos in the dataloader, not the frames or instances
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    abnormal_train_loader = data.DataLoader(
        FeatureDataset(
            data_dir=config.root_dir,
            mode="Train",
            modal=config.modal,
            num_segments=config.num_segments,
            len_feature=config.len_feature,
            is_normal=False,
        ),
        batch_size=4,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        FeatureDataset(
            data_dir=config.root_dir,
            mode="Test",
            modal=config.modal,
            num_segments=config.num_segments,
            len_feature=config.len_feature,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
    )
    # len(dataloader) = num_videos/batch_size
    # print(len(normal_train_loader))
    # print(len(abnormal_train_loader))
    print(len(test_loader))

    test_info = {"step": [], "auc": [], "ap": [], "ac": []}

    best_auc = 0

    criterion = AD_Loss()  # specific loss function used by the code

    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.lr[0], betas=(0.9, 0.999), weight_decay=0.00005
    )

    train_loss = []
    test_metrics = []

    test(net, test_loader, test_info, step=0, args=args)

    for step in tqdm(
        range(1, config.num_iters + 1), total=config.num_iters, dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
        train_cost = train(
            net,
            normal_loader_iter,
            abnormal_loader_iter,
            optimizer,
            criterion,
            step,
        )
        train_loss.append(train_cost)

        if step % 10 == 0 and step > 10:
            # NEW
            test(net, test_loader, test_info, step=step, args=args)
            current_auc = test_info["auc"][-1]
            current_ap = test_info["ap"][-1]
            test_metrics.append([step, current_auc, current_ap])
            if test_info["auc"][-1] > best_auc:
                # best_auc = test_info["auc"][-1]
                # utils.save_best_record(
                #     test_info,
                #     os.path.join(
                #         config.output_path,
                #         "ur_dmu_best{}.txt".format(config.seed),
                #     ),
                # )

                torch.save(
                    net.state_dict(),
                    os.path.join(
                        args.model_path, "ur_dmu_best{}.pkl".format(config.seed)
                    ),
                )
            # if step == config.num_iters:
            #     torch.save(
            #         net.state_dict(),
            #         os.path.join(args.model_path, "moerdijk_trans_{}.pkl".format(step)),
            #     )

    train_costs_np = np.array(train_loss)

    save_path = os.path.join(config.output_path, "train_costs.csv")
    np.savetxt(save_path, train_costs_np, delimiter=",")
    test_metrics_df = pd.DataFrame(test_metrics, columns=["step", "auc", "ap"])
    save_metrics_path = os.path.join(config.output_path, "test_metrics.csv")
    test_metrics_df.to_csv(save_metrics_path, index=False)
