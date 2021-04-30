python -m semantic_matching.encoder \
--encoder siamese_cbow \
--max_len 100 \
--pooling mean \
--batch_size 32 \
--data_file data/train.txt \
--save_dir output \
--epochs 5 \
--device cpu \
--workers 1