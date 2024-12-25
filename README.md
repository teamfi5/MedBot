# MedBot
MedBot là một ứng dụng chatbot sử dụng mô hình học sâu để trả lời các câu hỏi về y tế. Dưới đây là hướng dẫn cài đặt và sử dụng ứng dụng này.

## Hướng dẫn cài đặt
### 1. Clone repository về máy:

```bash
Sao chép mã
git clone https://github.com/teamfi5/MedBot
```
### 2. Tải mô hình và dữ liệu cần thiết:

Bạn có thể tải mô hình và dữ liệu cần thiết tại [Đây](link-to-model).
Hoặc, nếu muốn tinh chỉnh lại mô hình, hãy làm theo các bước sau:
#### Bước 1: Tải hoặc tạo dữ liệu huấn luyện
Tải thư mục finetune_data từ [Đây](link) và đặt nó vào thư mục dự án của bạn.

Hoặc, tự tạo dữ liệu huấn luyện theo cấu trúc và định dạng mẫu sau:
Cấu trúc:
```markdown
project/
├── finetune_data/
    ├── embedding/
    │   ├── train_embed.jsonl
    │   └── test_embed.jsonl
    ├── llm/
    │   └── qs_ans.jsonl
    ├── rerank/
    │   ├── pair_score.csv
    │   └── test_pair_score.csv
```
Định dạng mẫu:
train_embed.jsonl và test_embed.jsonl:
```bash
{"query": str,"pos":list(str),"neg":list(str)}
```
qs_ans.jsonl:
```bash
{"context": str, "output": str, "input": str}
```

pair_score.csv và test_pair_score.csv
```bash
collumn('sentence1'):str, collumn('sentence2'):str, collumn('score'):float
```

#### Bước 2: Huấn luyện mô hình
Chạy các file sau để huấn luyện mô hình và lưu lại kết quả:

finetune_reranker.ipynb: Huấn luyện mô hình reranker.
finetune_llm.ipynb: Huấn luyện mô hình LLM (Large Language Model).
Chạy file create_vectorDB.ipynb để tạo vector database từ dữ liệu đã huấn luyện.

Bước 3: Kiểm tra hệ thống
Chạy file test.ipynb để kiểm tra hoạt động của hệ thống sau khi huấn luyện và tạo vector database.
Sau khi hoàn tất các bước trên, bạn có thể chạy ứng dụng với lệnh sau:

```bash
python app.py
```
Truy cập ứng dụng qua trình duyệt tại http://127.0.0.1:5000.



         
   
                   
