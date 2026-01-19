"""
E-MDAI Training Processor (Entropy-Guided Modality Decoupling & Alignment Intervention)

针对非平衡多模态学习的干预机制：
利用信息熵监控模态主导地位，并在必要时对弱势模态进行“正交梯度干预”。
"""

import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
from layers.iadd import IADDPlugin, HybridTripletLoss
from layers.emdai import EMDAIPlugin

def do_train_emdai(cfg,
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
    logger.info('start training with E-MDAI Intervention')

    num_classes = model.num_classes if hasattr(model, 'num_classes') else None

    # IADD 插件仍可作为基础组件保留
    use_iadd = cfg.MODEL.IADD.ENABLED
    if use_iadd:
        iadd_plugin = IADDPlugin(
            temperature=cfg.MODEL.IADD.TEMPERATURE,
            hard_neg_k=cfg.MODEL.IADD.HARD_NEG_K,
            lambda_distill=cfg.MODEL.IADD.LAMBDA_DISTILL,
            lambda_hybrid=cfg.MODEL.IADD.LAMBDA_HYBRID
        ).to(device)
        hybrid_triplet_loss = HybridTripletLoss(margin=cfg.SOLVER.MARGIN).to(device)

    # E-MDAI 超参数
    # 你可以在 cfg 中添加这些，这里先使用默认建议值
    emdai_threshold = 0.4  # 熵阈值
    
    # 初始化 E-MDAI 插件
    logger.info(f"Initializing E-MDAI Plugin with threshold={emdai_threshold}...")
    emdai_plugin = EMDAIPlugin(threshold=emdai_threshold, num_classes=num_classes).to(device)
    logger.info("E-MDAI Plugin initialized.")
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    mcd_rgb_meter = AverageMeter()
    mcd_ir_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

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
                # 1. 前向传播
                output = model(img, label=target, cam_label=target_cam, view_label=target_view)
                
                loss_total = 0
                
                if isinstance(output, dict):
                    # 获取核心特征和 Logits
                    score_final = output.get('moe_score', output.get('ori_score'))
                    feat_final = output.get('moe_feat', output.get('ori_feat'))
                    
                    # 提取各模态组件
                    logits_dict = output['logits_dict']
                    feats_dict = output['feats_dict']
                    
                    rgb_logits = logits_dict['RGB']
                    rgb_feats = feats_dict['RGB']
                    ni_feats = feats_dict['NI']
                    ti_feats = feats_dict['TI']
                    
                    # 2. 执行 E-MDAI 干预 (Plugin 调用)
                    # 我们分别对 NI 和 TI 进行干预，以 RGB 为基准
                    ni_feats_intervened, ni_stats = emdai_plugin(ni_feats, rgb_feats, rgb_logits)
                    ti_feats_intervened, ti_stats = emdai_plugin(ti_feats, rgb_feats, rgb_logits)
                    
                    # 3. 重新构建用于后续 Loss 计算的“受干预”特征
                    # 注意：我们只需干预参与核心 Loss 计算的特征流
                    # 这里我们模拟 IR 融合逻辑
                    ir_feats_intervened = (ni_feats_intervened + ti_feats_intervened) / 2
                    ir_logits = (logits_dict['NI'] + logits_dict['TI']) / 2
                    
                    # 4. 计算 Loss
                    # 基础 Loss
                    loss_base = loss_fn(score=score_final, feat=feat_final, 
                                      target=target, target_cam=target_cam)
                    loss_total += loss_base
                    
                    # IADD 组件（如果启用，使用干预后的特征）
                    if use_iadd:
                        iadd_out = iadd_plugin(
                            rgb_logits, ir_logits,
                            rgb_feats, ir_feats_intervened,
                            target
                        )
                        loss_total += iadd_out['loss_distill']
                        loss_hybrid = hybrid_triplet_loss(iadd_out['hybrid_dist'], target)
                        loss_total += loss_hybrid * cfg.MODEL.IADD.LAMBDA_HYBRID
                        
                        mcd_rgb_meter.update(iadd_out['mcd_m1'], 1)
                        mcd_ir_meter.update(iadd_out['mcd_m2'], 1)

                else:
                    # 非字典输出的兼容逻辑（略）
                    score_final = output[0]
                    loss_total += loss_fn(score=output[0], feat=output[1], target=target, target_cam=target_cam)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (score_final.max(1)[1] == target).float().mean()
            acc_meter.update(acc, 1)
            loss_meter.update(loss_total.item(), img['RGB'].shape[0])

            if (n_iter + 1) % log_period == 0:
                # 在日志中增加干预比例的显示，方便观察 E-MDAI 是否工作
                interv_ratio = (ni_stats['intervention_ratio'] + ti_stats['intervention_ratio']) / 2
                logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f} Acc: {:.3f} Interv_Ratio: {:.2f} lr: {:.2e}"
                          .format(epoch, (n_iter + 1), len(train_loader),
                                 loss_meter.avg, acc_meter.avg, interv_ratio,
                                 scheduler._get_lr(epoch)[0]))

    return best_index

def training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=1):
    from engine.processor import training_neat_eval as eval_func
    return eval_func(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern)
