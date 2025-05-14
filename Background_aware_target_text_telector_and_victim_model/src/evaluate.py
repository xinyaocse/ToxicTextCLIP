import os
import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .scheduler import cosine_scheduler
from sklearn.metrics.pairwise import cosine_similarity


def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader,ncols=120):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking=True), batch[
                "attention_mask"].to(options.device, non_blocking=True), batch["pixel_values"].to(options.device,
                                                                                                  non_blocking=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

            umodel = model.module if (options.distributed) else model

            logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
            logits_per_text = logits_per_image.t()

            target = torch.arange(len(input_ids)).long().to(options.device, non_blocking=True)
            loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["loss"] = loss

    logging.info("Finished validating")

    return metrics


def get_zeroshot_metrics(model, processor, test_dataloader, options):
    logging.info("Started zeroshot testing")

    model.eval()
    umodel = model.module if (options.distributed) else model

    config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]
    with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes):
            text = [template(c) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens[
                "attention_mask"].to(options.device)
            text_embedding = umodel.get_text_features(input_ids=text_input_ids, attention_mask=text_attention_mask)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim=1).to(options.device)

    with torch.no_grad():
        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}

        for image, label in tqdm(test_dataloader):
            image = image.to(options.device)
            label = label.to(options.device)
            # image, label = image.to(options.device), label.to(options.device)
            image_embedding = umodel.get_image_features(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            logits = (image_embedding @ text_embeddings)
            ranks = logits.topk(max(topk), 1)[1].T
            predictions = ranks == label

            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim=0)).item()

    results = {f"zeroshot_top{k}": correct[k] / test_dataloader.num_samples for k in topk}
    logging.info("Finished zeroshot testing")

    return results



def evaluate(epoch, model, processor, data, options):
    metrics = {}

    if (options.master):
        if (data["validation"] is not None or data["eval_test"] is not None):
            if (epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if (data["validation"] is not None):
            metrics.update(get_validation_metrics(model, data["validation"], options))

        if (metrics):
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if (options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics

