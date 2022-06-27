python infer_swintransformer_plugin.py \
    --eval \
    --cfg ../../../swin_tiny_patch4_window7_224_lite.yaml \
    --resume ../../../best.pth \
    --th-path ../../../build/lib/libpyt_swintransformer.so \
    --engine swin_transformer_fp32.engine \
    --batch-size 1 
