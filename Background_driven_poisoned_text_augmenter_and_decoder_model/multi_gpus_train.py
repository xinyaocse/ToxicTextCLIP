import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW

import clip
import models
from clip.simple_tokenizer import SimpleTokenizer as Tokenizer
from src.parser import parse_args
from utils import utils
from utils.constants import *
from utils.data_utils import get_data as data_load
from utils.data_utils import (
    get_masks_and_count_tokens_trg,
    get_src_and_trg_batches,
    split_text_and_trgtext,
)
from utils.optimizers_and_distributions import (
    CustomLRAdamOptimizer,
    LabelSmoothingDistribution,
)

mp.set_start_method("spawn", force=True)

# Simple decorator function so that I don't have to pass these arguments every time I call get_train_val_loop
def get_train_val_loop(
    model,
    optimizer,
    custom_lr_scheduler,
    kl_div_loss,
    label_smoothing,
    pad_token_id,
    time_start,
    scaler,
    logger_=None,
):
    def train_val_loop(options_, is_train, token_ids_data, epoch):
        if logger_ is not None:
            logger_output = logger_.bind(name="output")
        if is_train:
            model.train()
            # for name, param in baseline_transformer.named_parameters():
            #     if param.requires_grad == False:
            #         print(f"param {name} not requires_grad")
            if options_.distributed:
                token_ids_data.set_epoch(epoch)
        else:
            model.eval()

        # use_baseline_transformer = baseline_transformer.module if(options.distributed) else baseline_transformer

        device = next(model.parameters()).device
        token_ids_loader = token_ids_data.dataloader
        if logger_ is not None and options_.master:
            # logger_output.info(f"Num samples: {token_ids_loader.num_samples}, Num_batches: {token_ids_loader.num_batches}")
            logger_output.info(
                f"Num samples: {token_ids_loader.num_samples}, Num_batches: {token_ids_loader.num_batches}"
            )
            # logger_output.info(f"Num samples: {token_ids_loader.num_samples}")

        #
        # Main loop - start of the CORE PART
        #
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):

            image_tensor, all_text_tokens = token_ids_batch[0], token_ids_batch[1]
            text_tokens, text_target_tokens = split_text_and_trgtext(all_text_tokens)

            image_tensor, text_tokens, text_target_tokens = (
                image_tensor.to(device),
                text_tokens.to(device),
                text_target_tokens.to(device),
            )

            (
                src_token_ids_batch,
                trg_token_ids_batch_input,
                trg_token_ids_batch_gt,
            ) = get_src_and_trg_batches(text_tokens, text_target_tokens)

            # get trg_mask
            trg_mask = get_masks_and_count_tokens_trg(
                trg_token_ids_batch_input, pad_token_id
            )
            src_mask = None

            with autocast(device_type="cuda"):
                # log because the KL loss expects log probabilities (just an implementation detail)
                predicted_log_distributions = model(
                    pixel_values=image_tensor,
                    text_values=text_tokens,
                    trg_token_ids_batch=trg_token_ids_batch_input,
                    trg_mask=trg_mask,
                    src_mask=src_mask,
                )

                smooth_target_distributions = label_smoothing(
                    trg_token_ids_batch_gt.long()
                )  # these are regular probabilities

                loss = kl_div_loss(
                    predicted_log_distributions, smooth_target_distributions
                )

            if is_train:

                custom_lr_scheduler.step()
                optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph

                scaler.scale(
                    loss
                ).backward()  # compute the gradients for every trainable weight in the computational graph
                scaler.step(optimizer)  # apply the gradients to weights
                scaler.update()

            # End of CORE PART

            #
            # Logging and metrics
            #
            if options_.master and is_train:
                if options_.console_log_freq is not None and (
                    batch_idx % options_.console_log_freq == 0
                    or (batch_idx == token_ids_loader.num_batches - 1)
                ):
                    num_samples = (
                        (batch_idx + 1) * len(image_tensor) * options_.num_devices
                    )
                    logger_output.info(
                        f"Train Epoch: {epoch} [{num_samples}/{token_ids_loader.num_samples} ({100.0 * (batch_idx + 1) / token_ids_loader.num_batches:.0f}%)]\t\tLoss: {loss.item():.6f}\tTime taken {time.time() - time_start:.2f}[s]"
                    )
            elif options_.master:
                if options_.console_log_freq is not None and (
                    batch_idx % options_.console_log_freq == 0
                    or (batch_idx == token_ids_loader.num_batches - 1)
                ):
                    num_samples = (batch_idx + 1) * len(image_tensor)
                    logger_output.info(
                        f"Val Epoch: {epoch} [{num_samples} ({100.0 * (batch_idx + 1) / token_ids_loader.num_batches:.0f}%)]\t\tLoss: {loss.item():.6f}\tTime taken {time.time() - time_start:.2f}[s]"
                    )

    return train_val_loop


def train_model(rank, options_, logger_):

    pure_tokenize = Tokenizer()

    options_.local_rank = rank
    options_.master = rank == 0

    logger_output = logger_.bind(name="output")
    logger_params = logger_.bind(name="params")

    if options_.device == "cuda":
        options_.device += ":" + str(
            options_.device_ids[options_.local_rank]
            if options_.distributed
            else options_.device_id
        )

    logger_output.info("Use {} device", options_.device)

    if options_.master:
        logger_params.info("Params:")
        for key in sorted(vars(options_)):
            value = getattr(options_, key)
            logger_params.info(f"{key}: {value}")

    # pad_token_id = Tokenizer().encode(PAD_TOKEN)  # pad token id is the same for target as well

    cudnn.benchmark = True
    cudnn.deterministic = False

    if options_.distributed:
        dist.init_process_group(
            backend=options_.distributed_backend,
            init_method=options_.distributed_init_method,
            world_size=options_.num_devices,
            rank=rank,
        )

    # Step 2: Prepare the model and push to GPU

    model = models.load_my_model(options_)

    if options_.master:
        logger_params.info("CLIP parameters frozen situation:")
        for name, param in model.clip_model.named_parameters():
            logger_params.info(f"{name} requires_grad: {param.requires_grad}")

        logger_params.info("Model structure:")
        logger_params.info(str(model))

    torch.cuda.set_device(
        options_.device_ids[options_.local_rank]
        if options_.distributed
        else options_.device_id
    )
    model.to(options_.device)

    if options_.distributed:
        # DDP
        model = DDP(model, device_ids=[options_.device_ids[options_.local_rank]])
    module = model.module if options_.distributed else model
    # Step 3: Prepare data loaders

    options_.batch_size = options_.batch_size // options_.num_devices

    load_start = time.time()
    data1 = data_load(
        args=options_,
        preprocess_image=module.image_preprocess,
        epoch=0,
        tokenizer=clip.tokenize,
        pad_token_id=options_.pad_token_id,
    )
    load_end = time.time()
    if options_.master:
        logger_output.info("data loading time: {}".format(str(load_end - load_start)))
    # train_token_ids_data and val_token_ids_data is DataInfo are in DataInfo format. To obtain DataLoader,. dataLoader is required
    train_token_ids_data, val_token_ids_data = data1["train"], data1["val"]

    # Step 4: Prepare other training related utilities
    kl_div_loss = nn.KLDivLoss(
        reduction="batchmean"
    )  # gives better BLEU score than "mean"

    # Makes smooth target distributions as opposed to conventional one-hot distributions
    # My feeling is that this is a really dummy and arbitrary heuristic but time will tell.
    label_smoothing = LabelSmoothingDistribution(
        options_.label_smoothing_value,
        options_.pad_token_id,
        trg_vocab_size=module.clip_model.vocab_size,
        device=options_.device,
    )

    optimizer = Adam(model.parameters(), betas=(0.9, 0.98), eps=1.0e-6)

    # Check out playground.py for an intuitive visualization of how the LR changes with time/training steps, easy stuff.
    custom_lr_scheduler = CustomLRAdamOptimizer(
        optimizer=optimizer,
        model_dimension=module.clip_model.ln_final.weight.shape[0],
        num_of_warmup_steps=options_.num_warmup_steps,
    )

    scaler = GradScaler()

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    train_val_loop = get_train_val_loop(
        model,
        optimizer,
        custom_lr_scheduler,
        kl_div_loss,
        label_smoothing,
        options_.pad_token_id,
        time.time(),
        scaler=scaler,
        logger_=logger_,
    )

    # Step 5: Start the training
    for epoch in range(options_.num_of_epochs):
        if options_.master:
            logger_output.info(f"Starting Epoch {epoch}")

        # Training loop
        start = time.time()
        train_val_loop(
            options_=options_,
            is_train=True,
            token_ids_data=train_token_ids_data,
            epoch=epoch,
        )
        end = time.time()

        if options_.master:
            logger_output.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

        # Validation loop
        with torch.no_grad():
            if options_.master:
                logger_output.info(f"Epoch {epoch} starting validating.")
            start = time.time()
            train_val_loop(
                options_=options_,
                is_train=False,
                token_ids_data=val_token_ids_data,
                epoch=epoch,
            )

            if options_.master:
                logger_output.info(
                    f"Epoch {epoch} validating finish, time taken: {time.time() - start:.3f}."
                )

        if options_.master:
            # Save model checkpoint
            if (
                options_.checkpoint_freq is not None
                and (epoch + 1) % options_.checkpoint_freq == 0
            ):
                ckpt_model_name = f"epoch_{epoch + 1}.pth"
                torch.save(
                    utils.get_training_state(options_, module),
                    os.path.join(CHECKPOINTS_PATH, ckpt_model_name),
                )

    if options_.master:
        # Save the latest transformer in the binaries directory
        torch.save(
            utils.get_training_state(options_, module),
            os.path.join(BINARIES_PATH, utils.get_available_binary_name(BINARIES_PATH)),
        )

    logger_.complete()

    if options_.distributed:
        dist.destroy_process_group()


def output_filter(record):
    return record["extra"]["name"] == "output"


def parems_filter(record):
    return record["extra"]["name"] == "params"


if __name__ == "__main__":

    options = parse_args()

    options.pad_token_id = 400

    logger.remove()
    logger.add(
        "log/output_{time}.log",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {process} : {message}",
        filter=output_filter,
    )
    logger.add(
        "log/params_{time}.log", enqueue=True, format=" {message}", filter=parems_filter
    )
    logger.add(
        sink=sys.stdout,
        enqueue=True,
        colorize=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {process} : {message}",
        filter=output_filter,
    )

    ngpus = torch.cuda.device_count()

    options.device = "cuda"
    if options.device_ids is None:
        options.device_ids = list(range(ngpus))
        options.num_devices = ngpus
    else:
        options.device_ids = list(map(int, options.device_ids))
        options.num_devices = len(options.device_ids)

    options.distributed = True

    mp.spawn(train_model, nprocs=options.num_devices, args=(options, logger,))
