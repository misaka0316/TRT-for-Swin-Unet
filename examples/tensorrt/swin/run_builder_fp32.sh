python builder_fp32.py \
    --batch-size 32 \
    --cfg ../../../swin_tiny_patch4_window7_224_lite.yaml \
    --resume ../../../best.pth \
    --th-path ../../../build/lib/libpyt_swintransformer.so \
    --output swin_transformer_fp32.engine

