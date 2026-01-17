#!/bin/bash
# 多模态消融实验快速启动脚本

# 设置默认参数
CONFIG_FILE="configs/RGBNT201/vit_demo.yml"
WARMUP_EPOCHS=10
GPU_IDS="0,1"
OUTPUT_DIR="logs/RGBNT201/modality_ablation"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP_EPOCHS="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_IDS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -c, --config FILE      配置文件路径 (default: configs/RGBNT201/vit_demo.yml)"
            echo "  -w, --warmup EPOCHS    Warm-up轮数 (default: 10)"
            echo "  -g, --gpu IDS          GPU设备ID (default: 0,1)"
            echo "  -o, --output DIR       输出目录 (default: logs/RGBNT201/modality_ablation)"
            echo "  -h, --help             显示帮助信息"
            echo ""
            echo "Example:"
            echo "  $0 -c configs/RGBNT201/vit_demo.yml -w 15 -g 0,1,2,3"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# 显示配置信息
echo "========================================="
echo "多模态消融实验"
echo "========================================="
echo "配置文件: $CONFIG_FILE"
echo "Warm-up轮数: $WARMUP_EPOCHS"
echo "GPU设备: $GPU_IDS"
echo "输出目录: $OUTPUT_DIR"
echo "========================================="
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 开始训练
echo "开始训练..."
echo ""

python train_modality_ablation.py \
    --config_file "$CONFIG_FILE" \
    --warmup_epochs $WARMUP_EPOCHS \
    MODEL.DEVICE_ID "$GPU_IDS" \
    OUTPUT_DIR "$OUTPUT_DIR"

TRAIN_EXIT_CODE=$?

# 检查训练是否成功
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "训练失败，退出码: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "========================================="
echo "训练完成！"
echo "========================================="
echo ""

# 询问是否进行结果分析
read -p "是否立即分析结果？(y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "========================================="
    echo "开始分析结果..."
    echo "========================================="
    echo ""

    python analyze_modality_results.py --result_dir "$OUTPUT_DIR"

    ANALYSIS_EXIT_CODE=$?

    if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "分析完成！"
        echo "========================================="
        echo ""
        echo "生成的文件："
        echo "1. $OUTPUT_DIR/modality_ablation_results.json"
        echo "2. $OUTPUT_DIR/modality_impact_summary.txt"
        echo "3. $OUTPUT_DIR/modality_performance_curves.png"
        echo "4. $OUTPUT_DIR/modality_importance.png"
        echo "5. $OUTPUT_DIR/detailed_analysis_report.txt"
        echo ""
        echo "查看摘要报告："
        echo "cat $OUTPUT_DIR/modality_impact_summary.txt"
        echo ""
    else
        echo "分析失败，退出码: $ANALYSIS_EXIT_CODE"
        exit $ANALYSIS_EXIT_CODE
    fi
fi

echo "完成！"
