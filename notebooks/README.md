# Notebooks

## Notebook chuẩn

- `extract-kf-transnetv2.ipynb`
- `ingest-asr-chunkformer-rnnt-large.ipynb`
- `ingest-od-wedetect-large.ipynb`
- `ingest-visual-siglip2-so400m.ipynb`
- `ingest-visual-beit3-large-coco-retrieval.ipynb`

Các file này là bản nguồn sạch để upload lên Colab/Kaggle. Chúng không phải bằng
chứng rằng kiến trúc retrieval hoặc model ensemble đã được quyết định.

## Notebook đã chạy

Đặt bản download có output vào `runs/`. Git bỏ qua các file `.ipynb` ở đó để
tránh lưu log lớn hoặc dữ liệu nhạy cảm. Có thể chạy
`python scripts/strip_notebook_outputs.py <file>` trên một bản sao trước khi
đưa thay đổi logic trở lại `ingestion/`.
