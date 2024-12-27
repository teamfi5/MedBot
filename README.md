# MedBot
MedBot là một ứng dụng chatbot sử dụng kĩ thuật RAG retrieve để truy xuất tài liệu và LLM để trả lời các câu hỏi về y tế. Dưới đây là hướng dẫn cài đặt và sử dụng ứng dụng này.

# Link github
```bash
https://github.com/teamfi5/MedBot
```

## Kiến trúc hệ thống

<a href="https://ibb.co/Jm4rkV4"><img src="https://i.ibb.co/QpRYmLR/ktht.jpg" alt="ktht" border="0" /></a>

## Kết quả thực nghiệm
### Truy xuất

| Metric      | Value   |
|-------------|---------|
| MRR@10      | 0.7762  |
| Recall@10   | 0.9286  |

Kết quả cho thấy context đúng có tỉ lệ 92.86% xuất hiện trong 10 contexts đầu tiên và trung bình xuất hiện ở vị trí 1 / 0.7762 ≈ 1.3 (Vị trí cao trong tập kết quả truy xuất)

### Xếp hạng lại
MRR@10 = 0.8032: Đã cải thiện so với bộ truy xuất ban đầu

<a href="https://drive.google.com/file/d/1ngpt1qqe29o15pV9QnX-KYWoQVsopeQ2/view?usp=drive_link"><img src="https://drive.google.com/file/d/1ngpt1qqe29o15pV9QnX-KYWoQVsopeQ2/view?usp=drive_link" alt="rrl" border="0" /></a>

### Sinh văn bản

<a href="https://ibb.co/gdNvKHM"><img src="https://i.ibb.co/xYWmTPS/llml.jpg" alt="llml" border="0" /></a>

Thử nghiệm:
```bash
context = """
Dấu hiệu và triệu chứng \nNhững dấu hiệu và triệu chứng có thể là của ung thư phổi bao gồm:\nTriệu chứng về đường hô hấp: ho, ho ra máu, thở khò khè, khó thở\nTriệu chứng toàn thân: sụt cân, mệt mỏi, sốt, móng tay dùi trống\nTriệu chứng do ung thư chèn ép nhiều sang các cơ quan kề bên: đau ngực, đau xương, tắc nghẽn tĩnh mạch chủ trên, khó nuốt\nNếu ung thư phát triển ở đường thở, nó có thể chặn dòng khí lưu thông, gây ra chứng khó thở. Sự cản trở này có thể dẫn tới việc tích lũy chất bài tiết phía sau chỗ tắc, qua đó mở đường cho viêm phổi.'
Ung thư phổi là căn bệnh trong đó xuất hiện một khối u ác tính được mô tả qua sự tăng sinh tế bào không thể kiểm soát trong các mô phổi. Nếu người bệnh không được điều trị, sự tăng trưởng tế bào  này có thể lan ra ngoài phổi  đến các mô hoặc bộ phận khác của cơ thể, quá trình này gọi là di căn. Hầu hết các loại ung thư khởi nguồn từ trong phổi (ung thư phổi nguyên phát) là ung thư biểu mô,. Ung thư phổi được chia làm hai loại chính là ung thư phổi tế bào nhỏ (SCLC) và ung thư phổi không phải tế bào nhỏ (NSCLC). Triệu chứng phổ biến nhất của căn bệnh này là ho (bao gồm cả ho ra máu), sụt cân, khó thở, và đau ngực.',
Phần lớn các triệu chứng của ung thư phổi (chán ăn, sụt cân, sốt, mệt mỏi) là không đặc biệt. Đối với nhiều người, vào thời điểm họ phát hiện ra những dấu hiệu bệnh tật và đi tìm sự chăm sóc y tế, khối u đã lan ra ngoài địa điểm khởi phát. Các triệu chứng có thể báo hiệu quá trình di căn đã xuất hiện bao gồm sụt cân, đau xương và các triệu chứng về thần kinh (đau đầu, ngất xỉu, co giật, yếu chi). Những địa điểm khối u lan sang thường gặp đó là não, xương, tuyến thượng thận, lá phổi còn lại, gan, màng ngoài tim, và thận. Khoảng 10% số ca ung thư phổi không thấy những triệu chứng khi chẩn đoán, những trường hợp này bệnh tình cờ phát hiện nhờ việc chụp X quang ngực định kỳ.\n Nguyên nhân'"""

query = """Triệu chứng của ung thư phổi?"""

from generation import QAmodel
Bot = QAmodel()

print(Bot.generation(query, context)
```
Kết quả:
```bash
Triệu chứng phổ biến nhất của ung thư phổi là ho, sụt cân, sốt, khó thở, và đau ngực. Ngoài ra, các triệu chứng có thể bao gồm sụt cân, đau xương, và các triệu chứng về thần kinh. Những địa điểm khối u lan sang thường gặp là não, xương, tuyến thượng thận, lá phổi còn lại, gan, màng ngoài tim, và thận. Khoảng 10% số ca ung thư phổi không thấy những triệu chứng khi chẩn đoán. Chẩn đoán ung thư phổi thường bắt đầu bằng chụp X quang ngực định kỳ. 
```


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



         
   
                   
