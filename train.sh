echo "Training VSQ"
python cmd_args_run.py \
    --model_name "VSQ" \
    --vector_decoder_model "mlp" \
    --quantized_dim 512 \
    --image_loss "mse" \
    --vq_method "fsq" \
    --fsq_levels 7 5 5 5 5 \
    --num_segments 3 \
    --num_codes_per_shape 2 \
    --alpha 0.05 \
    --dataset "figr8" \
    --csv_path ".data/stage1.csv" \
    --channels 3 \
    --width 128 \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --num_workers 16 \
    --individual_max_length 8.0 \
    --max_shapes_per_svg 32 \
    --lr 2.0e-05 \
    --train_log_interval 0.5 \
    --manual_seed 42 \
    --devices -1 \
    --max_epochs 1 \
    --val_check_interval 0.5 \
    --save_base_dir "results/VSQ_l" \
    --version 0 \
    --accumulate_grad_batches 2

echo "Tokenizing FIGR-8 for the TM"
python scripts/tokenize_svg_dataset.py

echo "Training TM"
TOKENIZERS_PARALLELISM=false python run_stage2.py --config configs/tm_l.yaml