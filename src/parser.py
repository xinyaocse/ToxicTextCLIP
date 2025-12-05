import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--console_log_freq",
        type=int,
        help="log to output console (batch) freq",
        default=10,
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        help="checkpoint model saving (epoch) freq",
        default=1,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Specify device type to use (default: gpu > cpu)",
    )

    parser.add_argument(
        "--device_id", type=int, default=0, help="Specify device id if using single gpu"
    )

    parser.add_argument(
        "--dropout_probability", type=float, help="dropout probability", default=0.1
    )
    parser.add_argument(
        "--label_smoothing_value", type=float, help="label smoothing value", default=0.1
    )
    parser.add_argument(
        "--CLIP_model_name",
        type=str,
        default="ViT-B/32",
        choices=["RN50x16", "ViT-B/32"],
        help="Entering the model name means using the corresponding pre trained model, or you can input the weight path of your own trained CLIP model.",
    )
    # !!!!!!!!!!!!!!!!!
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        help="target number of tokens in a src/trg batch",
        default=15000,
    )

    parser.add_argument(
        "--num_of_epochs", type=int, help="number of training epochs", default=32
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="target number of tokens in a src/trg batch",
        default=832,
    )


    parser.add_argument("--train_data", type=str, default="/root/use_data/cc12m/{00000..01242}.tar::/root/use_data/cc3m/{00000..00331}.tar",
                        help="path to train data, need the webdataset format")


    parser.add_argument(
        "--val_data",
        type=str,
        default="/root/use_data/cc3mval/{00000..00001}.tar",
        help="path to val data",
    )


    parser.add_argument(
        "--use_beam_search_decoding",
        type=bool,
        default=False,
        help="Whether to use beam search decoding, default to using greedy_decoding",
    )

    # Multi GPU training
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Use multiple gpus if available",
    )

    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )

    parser.add_argument(
        "--distributed_backend", type=str, default="nccl", help="Distributed backend"
    )
    parser.add_argument(
        "--distributed_init_method",
        type=str,
        default="tcp://127.0.0.1:29500",
        help="Distributed init method",
    )
    parser.add_argument(
        "--device_ids",
        nargs="+",
        default=None,
        help="Specify device ids if using multiple gpus",
    )

    parser.add_argument(
        "--My_model_name",
        type=str,
        default="image_as_context",
        choices=["image_as_context"],
    )

    parser.add_argument(
        "--dataset_resampled",
        type=bool,
        default=False,
        help="Whether to use sampling with replacement for webdataset shard selection.",
    )

    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        ),
    )

    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )

    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )

    options = parser.parse_args()
    return options
