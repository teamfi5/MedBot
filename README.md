# MedBot
MedBot là một ứng dụng chatbot sử dụng mô hình học sâu để trả lời các câu hỏi về y tế. Dưới đây là hướng dẫn cài đặt và sử dụng ứng dụng này.

## Hướng dẫn cài đặt
### 1. Clone repository về máy:

```bash
git clone https://github.com/teamfi5/MedBot
```

Tải các thư viện cần thiết trong file requirements.txt
```bash
accelerate==1.2.1
bitsandbytes==0.45.0
datasets==3.2.0
faiss-cpu==1.9.0.post1
Flask==3.1.0
Flask-Cors==5.0.0
huggingface-hub==0.27.0
idna==3.10
keras==3.7.0
langchain==0.3.13
langchain-community==0.3.13
langchain-huggingface==0.1.2
numpy==2.0.2
pandas==2.2.3
requests==2.32.3
scikit-learn==1.6.0
scipy==1.14.1
seaborn==0.13.2
sentence-transformers==3.3.1
tensorflow==2.18.0
tf_keras==2.18.0
torch==2.5.1+cu124
torchaudio==2.5.1+cu124
torchvision==0.20.1+cu124
transformers==4.47.1
trl==0.13.0
uvicorn==0.34.0
```
### 2. Tải mô hình và dữ liệu cần thiết:

Bạn có thể tải mô hình và dữ liệu cần thiết tại [Đây](https://drive.google.com/drive/folders/1m6Fvrng_7A3EnCEw5fDfU-gm1Ek_YAa8?usp=drive_link).
Hoặc, nếu muốn tinh chỉnh lại mô hình, hãy làm theo các bước sau:
#### Bước 1: Tải hoặc tạo dữ liệu huấn luyện
Tải thư mục finetune_data từ [Đây](https://drive.google.com/drive/folders/1flZiE7zGSTTW63ZBzhmMVe65R7h-o8PN?usp=sharing) và đặt nó vào thư mục dự án của bạn.

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
uvicorn app:app --reload
```
Truy cập ứng dụng qua trình duyệt tại [http://127.0.0.1:8000](http://127.0.0.1:8000/).



         
   
                   
