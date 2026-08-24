# Báo cáo thí nghiệm căn chỉnh mô hình theo dữ liệu sở thích

## 1. Phân tích và làm sạch dữ liệu

### Tổng quan quá trình nạp dữ liệu

- **Tổng số mẫu hợp lệ**: 24.
- **Lỗi được phát hiện**: dòng đầu tiên chứa dấu ngoặc kép chưa được escape quanh từ `self-attention`, khiến dòng này không phải JSON hợp lệ.
- **Các bước xử lý**:
  - Escape dấu ngoặc kép bị lỗi trong dữ liệu mẫu.
  - Bổ sung thông báo lỗi JSON/schema kèm tên file và số dòng.
  - Chuẩn hóa prompt để phát hiện prompt trùng lặp dù khác chữ hoa, chữ thường hoặc khoảng trắng.
  - Không chấp nhận các trường text rỗng hoặc chỉ chứa khoảng trắng.
  - Không chấp nhận cặp `chosen` và `rejected` giống hoặc gần giống nhau.
  - Bổ sung tùy chọn kiểm tra PII cơ bản cho địa chỉ email và số điện thoại.

### Chiến lược chia dữ liệu

- **Tỷ lệ train/validation thực tế**: 19/5 mẫu, xấp xỉ 80/20, sử dụng seed 42.
- **Cách ngăn data leakage**: các mẫu được gom nhóm theo prompt đã chuẩn hóa trước khi xáo trộn và chia tập. Vì vậy, toàn bộ mẫu có cùng prompt chỉ xuất hiện trong một tập, không thể đồng thời nằm trong cả train và validation.
- **Khả năng tái lập**: bộ sinh số ngẫu nhiên cục bộ với seed cố định giúp cùng dữ liệu và cấu hình luôn tạo ra cùng một cách chia.

## 2. Cài đặt DPO và ORPO

### Lựa chọn mục tiêu tối ưu

- **Phương pháp chính**: DPO, đúng với cấu hình mặc định trong `configs/local.yaml`.
- **Lý do lựa chọn**: DPO tối ưu trực tiếp mức ưu tiên giữa câu trả lời `chosen` và `rejected` so với một mô hình tham chiếu cố định, không cần huấn luyện reward model riêng.
- **Phần mở rộng**: ORPO cũng được cài đặt và kiểm thử để hỗ trợ đầy đủ cả hai lựa chọn của bài lab.

### Các siêu tham số chính

| Siêu tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `beta` | 0.1 | Điều khiển độ mạnh của tín hiệu preference trong DPO. |
| `lambda_orpo` | 0.1 | Trọng số của preference penalty trong ORPO. |
| `batch_size` | 2 | Số mẫu dự kiến xử lý trong một batch. |
| `max_length` | 512 | Độ dài token tối đa trong cấu hình training. |
| `steps` | 25 | Số bước tối ưu của CPU mock trainer. |
| `learning_rate` | 0.5 | Tốc độ cập nhật preference margin của mock trainer. |

### Đảm bảo ổn định số học

- **Vấn đề của DPO**: tính trực tiếp `-log(sigmoid(x))` có thể gây overflow hoặc underflow khi preference margin rất lớn.
- **Giải pháp cho DPO**: sử dụng `numpy.logaddexp(0, -x)`, tương đương với softplus và ổn định hơn về mặt số học.
- **Vấn đề của ORPO**: công thức log-odds cần tính `log(1 - exp(logp))`, không ổn định khi xác suất tiến gần 1.
- **Giải pháp cho ORPO**: sử dụng kỹ thuật `log1mexp`, chia miền tính toán giữa `log1p` và `expm1`, đồng thời clip biên xác suất bằng machine epsilon để tránh `NaN`.
- **Kiểm tra đầu vào**: các hàm loss kiểm tra batch một chiều, không rỗng, cùng shape, chỉ chứa số hữu hạn, log-probability không dương và siêu tham số nằm trong miền hợp lệ.

## 3. Triển khai CPU mock trainer

Repository không kèm model weights hoặc tokenizer, do đó bài sử dụng phương án mock trainer chạy trên CPU theo yêu cầu cho phép của lab.

Mock trainer tối ưu một preference margin vô hướng:

```text
margin = logp(chosen) - logp(rejected)
```

Gradient được xấp xỉ bằng central finite difference, sau đó margin được cập nhật bằng gradient descent. Mục tiêu của phần này là chứng minh loss được cài đặt có thể giảm theo đúng hướng, không phải fine-tune một Transformer thực tế.

Kết quả sau 25 bước:

| Chỉ số | Giá trị |
|---|---:|
| Loss DPO ban đầu | 0.6931 |
| Loss DPO cuối | 0.6628 |
| Preference margin cuối | 0.6157 |

Loss giảm và preference margin trở thành số dương, cho thấy quy trình tối ưu mô phỏng đi đúng hướng.

## 4. Phương pháp và kết quả đánh giá

### Phương pháp đánh giá

Do chưa có mô hình sinh thực tế, evaluator sử dụng một lexical scorer xác định và chạy được trên CPU. Scorer không đọc nhãn `chosen`/`rejected`; nó tính điểm dựa trên độ đa dạng token của câu trả lời và mức giao nhau từ vựng với prompt.

Pairwise accuracy được tính như sau:

```text
(số cặp chosen thắng + 0.5 × số cặp hòa) / tổng số cặp
```

### Kết quả trên validation

| Metric | Giá trị |
|---|---:|
| Số mẫu train | 19 |
| Số mẫu validation | 5 |
| Pairwise accuracy | 1.0000 |
| Mean preference margin | 0.4255 |

### Đánh giá định tính

- **Prompt**: “What is the difference between precision and recall?”
- **Điểm câu chosen**: 3.2718.
- **Điểm câu rejected**: 2.8663.
- **Kết quả**: scorer xếp câu chosen cao hơn câu rejected, phù hợp với nhãn của dữ liệu.
- **Cách diễn giải đúng**: kết quả cho thấy lexical baseline hoạt động trên cặp mẫu này, nhưng không chứng minh hệ thống hiểu chính xác kiến thức về precision và recall.

## 5. Kiểm thử và kiểm soát chất lượng

Các checkpoint đã chạy thành công:

| Checkpoint | Kết quả |
|---|---|
| `pytest -q` | 22 test pass |
| `ruff check src tests` | Pass |
| `mypy src` với strict mode | Pass |
| `pref-lab validate` | Nạp thành công 24 mẫu |
| `pref-lab train` | Loss giảm từ 0.6931 xuống 0.6628 |
| `pref-lab evaluate` | Ghi metrics JSON thành công |
| `git diff --check` | Pass; chỉ có cảnh báo quy ước LF/CRLF của Windows |

Test bao phủ các trường hợp chính: JSON lỗi có số dòng, prompt trùng sau chuẩn hóa, PII guard tùy chọn, split chống leakage, chosen/rejected gần trùng, công thức DPO/ORPO, extreme margin, shape không hợp lệ, tie trong pairwise accuracy và loss của mock trainer giảm.

## 6. Hạn chế và failure modes

- **Thiên lệch độ dài**: lexical scorer có xu hướng thích câu dài và nhiều từ khác nhau. Một câu dài nhưng sai vẫn có thể nhận điểm cao hơn câu ngắn nhưng đúng.
- **Validation nhỏ**: accuracy 1.0 chỉ được tính trên 5 mẫu; không thể xem đây là bằng chứng mô hình đạt chất lượng 100%.
- **Chưa fine-tune Transformer**: CPU trainer chỉ tối ưu một scalar minh họa, không cập nhật model weights và không tạo checkpoint mô hình.
- **PII guard chưa đầy đủ**: regex email/số điện thoại chỉ là lớp bảo vệ cơ bản, có thể bỏ sót hoặc báo nhầm.
- **Near-duplicate theo ký tự**: `SequenceMatcher` không phát hiện đầy đủ các câu paraphrase có cùng nghĩa.
- **Chưa chạy safety regression trước/sau**: repo không có mô hình sinh thực tế nên chưa thể so sánh câu trả lời trên bộ regression prompts trước và sau training.
- **Chưa có test set độc lập**: kết quả hiện tại chỉ sử dụng train/validation split.

## 7. Kết luận

Bài đã hoàn thiện các TODO của repository DPO/ORPO Alignment: validation dữ liệu, chia tập chống leakage, DPO/ORPO loss ổn định số học, CPU mock trainer, evaluator xác định, lưu metrics và kiểm thử tự động. Kết quả thực nghiệm chứng minh pipeline kỹ thuật hoạt động trong phạm vi CPU demo.

Tuy nhiên, kết quả không nên được diễn giải thành một mô hình preference-aligned hoàn chỉnh. Để triển khai thực tế cần tích hợp Transformer/TRL, tạo test set độc lập, chạy safety regression trước và sau training, đánh giá nội dung bằng người thật và quản lý model checkpoint ngoài Git.
