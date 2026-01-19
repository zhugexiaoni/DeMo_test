"""
IADD Training Processor

Training processor for IADD Plugin integration
"""

import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
from layers.iadd import IADDPlugin, HybridTripletLoss


def do_train_iadd(cfg,
                  model,
                  center_criterion,
                  train_loader,
                  val_loader,
                  optimizer,
                  optimizer_center,
                  scheduler,
                  loss_fn,
                  num_query,
                  local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("DeMo.train")
    logger.info('start training with IADD')

    # IADD 配置
    use_iadd = cfg.MODEL.IADD.ENABLED
    if use_iadd:
        logger.info("Initializing IADD Plugin...")
        iadd_plugin = IADDPlugin(
            temperature=cfg.MODEL.IADD.TEMPERATURE,
            hard_neg_k=cfg.MODEL.IADD.HARD_NEG_K,
            lambda_distill=cfg.MODEL.IADD.LAMBDA_DISTILL,
            lambda_hybrid=cfg.MODEL.IADD.LAMBDA_HYBRID
        ).to(device)
        hybrid_triplet_loss = HybridTripletLoss(margin=cfg.SOLVER.MARGIN).to(device)
        logger.info("IADD Plugin initialized.")

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    mcd_rgb_meter = AverageMeter()
    mcd_ir_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM

    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        mcd_rgb_meter.reset()
        mcd_ir_meter.reset()
        
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                # 假设模型返回字典，包含了 IADD 需要的组件
                output = model(img, label=target, cam_label=target_cam, view_label=target_view)
                
                loss_total = 0
                
                # --- 常规 Loss ---
                # 处理不同输出格式的兼容性
                if isinstance(output, dict):
                    # 使用融合后的特征计算基础 Loss
                    if 'moe_score' in output:
                        score_final = output['moe_score']
                        feat_final = output['moe_feat']
                    elif 'dmcg_score' in output and output['dmcg_score'] is not None:
                        score_final = output['dmcg_score']
                        feat_final = output['dmcg_feat']
                    else:
                        score_final = output.get('ori_score', output.get('score'))
                        feat_final = output.get('ori_feat', output.get('feat'))
                        
                    loss_base = loss_fn(score=score_final, feat=feat_final, 
                                      target=target, target_cam=target_cam)
                    loss_total += loss_base
                    
                    # --- IADD 插件逻辑 ---
                    if use_iadd:
                        # 防御式检查：如果缺少 IADD 所需字段，直接给出明确提示
                        if 'logits_dict' not in output or 'feats_dict' not in output:
                            logger.warning(
                                "IADD 已启用，但模型输出缺少 logits_dict 或 feats_dict，已跳过 IADD。"
                                "（常见原因：MODEL.DIRECT=1 时未返回 logits_dict）\n"
                                f"当前输出 keys={list(output.keys())}"
                            )
                        
                    if use_iadd and 'logits_dict' in output and 'feats_dict' in output:
                        # 提取特征和Logits
                        rgb_logits = output['logits_dict']['RGB']
                        rgb_feats = output['feats_dict']['RGB']
                        
                        # --- 三模态融合逻辑 ---
                        # 将 NI 和 TI 视为广义的 IR 模态
                        ni_logits = output['logits_dict']['NI']
                        ti_logits = output['logits_dict']['TI']
                        ni_feats = output['feats_dict']['NI']
                        ti_feats = output['feats_dict']['TI']
                        
                        # 简单的平均融合
                        ir_logits = (ni_logits + ti_logits) / 2
                        ir_feats = (ni_feats + ti_feats) / 2
                        # ---------------------
                        
                        iadd_out = iadd_plugin(
                            rgb_logits, ir_logits,
                            rgb_feats, ir_feats,
                            target
                        )
                        
                        # 1. 动态蒸馏 Loss
                        loss_total += iadd_out['loss_distill']
                        
                        # 2. 混合 Triplet Loss
                        loss_hybrid = hybrid_triplet_loss(iadd_out['hybrid_dist'], target)
                        loss_total += loss_hybrid * cfg.MODEL.IADD.LAMBDA_HYBRID
                        
                        # 记录
                        mcd_rgb_meter.update(iadd_out['mcd_m1'], 1)
                        mcd_ir_meter.update(iadd_out['mcd_m2'], 1)

                else:
                    # 兼容旧代码逻辑
                    if len(output) % 2 == 1:
                        loss_total += output[-1]
                        score_final = output[0]
                    else:
                        score_final = output[0]
                    
                    # 简单累加所有 loss
                    if isinstance(output, tuple) or isinstance(output, list):
                        for i in range(0, len(output)//2 * 2, 2):
                             loss_total += loss_fn(score=output[i], feat=output[i+1], target=target, target_cam=target_cam)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            # 计算准确率
            acc = (score_final.max(1)[1] == target).float().mean()
            acc_meter.update(acc, 1)
            loss_meter.update(loss_total.item(), img['RGB'].shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if use_iadd:
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f} Acc: {:.3f} MCD_RGB: {:.3f} MCD_IR: {:.3f} lr: {:.2e}"
                              .format(epoch, (n_iter + 1), len(train_loader),
                                     loss_meter.avg, acc_meter.avg, 
                                     mcd_rgb_meter.avg, mcd_ir_meter.avg,
                                     scheduler._get_lr(epoch)[0]))
                else:
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f} Acc: {:.3f} lr: {:.2e}"
                              .format(epoch, (n_iter + 1), len(train_loader),
                                     loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        # End of epoch
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Speed: {:.1f} samples/s".format(epoch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                             os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                         os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=3)
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                logger.info("Best mAP: {:.1%}, Rank-1: {:.1%}".format(best_index['mAP'], best_index['Rank-1']))

def training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=1):
    # 复用 processor.py 中的实现
    from engine.processor import training_neat_eval as eval_func
    return eval_func(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern)
