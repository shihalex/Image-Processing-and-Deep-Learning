# 影像處理與深度學習 | Image Processing and Deep Learning

本儲存庫整理「影像處理與深度學習」課程的五次作業、Jupyter Notebook 實作、實驗報告，以及以 **YOLOv13 進行交通號誌物件偵測**的期末專題報告。內容從 OpenCV 基礎影像操作、影像增強與傅立葉轉換，延伸至衛星影像分類、卷積自編碼器與物件偵測。

## 專案資訊

| 項目 | 內容 |
| --- | --- |
| 課程名稱 | 影像處理與深度學習（Image Processing and Deep Learning） |
| 開課單位 | 國立成功大學（NCKU）測量及空間資訊學系／研究所 |
| 開課學期 | 2025 Fall（114 學年度第 1 學期） |
| 專案性質 | 課程作業（Assignment 1–5）與期末專題 |
| 課程大綱 | [NCKU 課程大綱](https://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0114&sem=1&co_no=P681700&class_code=) |
| GitHub Repository | [GitHub Repository](https://github.com/shihalex/Image-Processing-and-Deep-Learning.git) |

## 目錄

- [專案資訊](#專案資訊)
- [各項作業說明](#各項作業說明)
- [期末專題-YOLOv13交通標誌偵測](#期末專題-YOLOv13交通標誌偵測)
- [參考資料](#參考資料)

## 各項作業說明

### Assignment 1 — 基礎影像處理

使用 OpenCV 與 Matplotlib 學習影像讀取、顯示與像素處理，包括：

- 了解 OpenCV 的 BGR 通道順序與 Matplotlib 的 RGB 顯示差異。
- 拆分 R、G、B 通道並比較像素強度顯示方式。
- 使用平均濾波進行模糊化、Canny 進行邊緣偵測，以及 OpenCV 範例中的圓形偵測。
- 使用 `cv2.threshold()` 實作二值化，並以 `cv2.calcHist()` 計算灰階直方圖。

### Assignment 2 — 影像增強

聚焦於灰階影像的亮度分布調整及空間域濾波：

- **Histogram Equalization：** 透過直方圖、機率分布與累積分布函數（CDF）進行灰階映射，改善對比度。
- **Histogram Matching：** 根據參考影像的累積分布建立映射，使來源影像的灰階分布接近目標分布。
- **Average Filter：** 透過 3×3 平均核平滑影像。
- **Sobel Filter：** 利用水平與垂直梯度核強化邊緣資訊。
- **2D Convolution：** 練習濾波器與影像的卷積運算。

報告亦說明離散灰階值與多對一映射，使直方圖等化後不一定呈現完全均勻分布的原因。

### Assignment 3 — 傅立葉轉換與頻域濾波

使用 NumPy FFT 觀察影像在空間域與頻率域之間的關係：

1. 以 `np.fft.fft2()` 計算二維離散傅立葉轉換。
2. 以 `np.fft.fftshift()` 將零頻率移至頻譜中心，顯示對數幅度頻譜。
3. 透過 `np.fft.ifftshift()` 與 `np.fft.ifft2()` 重建影像。
4. 設計理想低通及高通遮罩，觀察截止頻率（例如 5、30）對影像模糊與邊緣資訊的影響。

實驗顯示，低通濾波保留較平緩的影像變化，高通濾波則強調快速變化的邊緣與細節。

### Assignment 4 — EuroSAT 衛星影像分類

以 [EuroSAT](https://github.com/phelber/eurosat) 土地利用／土地覆蓋資料集進行十類影像分類，依序比較不同深度學習架構：

- **Model 1–3：** 不同深度及 Dropout 設計的全連接網路（FCN）。
- **Model 4–5：** 使用卷積、池化與分類層的 CNN。
- **Model 6：** 以預訓練 VGG16 進行遷移學習與微調。
- **Model 7–8：** 加入影像資料增強，以及 Early Stopping、ReduceLROnPlateau 等訓練策略。
- **進階模型：** Notebook 另包含 Model 9–10 的實驗段落，並探索 ResNet50、DenseNet121 等預訓練 Backbone。

Notebook 以線上下載的 EuroSAT RGB 影像進行實驗（原始下載網址可能隨時間變動）。課程報告記錄部分模型的測試準確率，例如基礎全連接網路約 **24.97%**、CNN Model 5 約 **83.01%**、VGG16 Model 6 約 **93.66%**。

### Assignment 5 — 卷積自編碼器影像去噪與去模糊

以 MNIST 手寫數字影像為例，建立卷積自編碼器（Convolutional Autoencoder）：

- **Encoder：** 以卷積層及 Max Pooling 壓縮影像特徵。
- **Decoder：** 以轉置卷積重建 28×28 單通道影像。
- **去噪（Denoising）：** 對輸入影像加入高斯雜訊，以原始乾淨影像作為訓練目標。
- **去模糊（Deblurring）：** 使用 OpenCV Gaussian Blur 產生模糊輸入，再學習重建原始影像。

報告記錄去噪實驗的最終訓練／驗證損失約 **0.0843 / 0.0840**，去模糊實驗約 **0.0640 / 0.0636**。

## 期末專題：YOLOv13 交通標誌偵測

期末專題研究 **YOLOv13 在道路交通標誌偵測上的應用**，比較直接微調預訓練模型與分階段遷移學習的訓練策略，並分析資料分布、辨識表現與訓練成本。

### 模型與資料

報告介紹 YOLOv13 的 Backbone、Neck、Head，以及 **HyperACE** 高階關係建模機制。研究資料由兩部分組成：

| 資料來源 | 影像數量 | 特性 |
| --- | ---: | --- |
| 實驗室資料集 | 17,390 | 主要為校園日間拍攝，類別分布相對均衡 |
| 自行蒐集資料集 | 3,397 | 真實道路環境影像，部分號誌類別樣本較少 |
| **合計** | **20,787** | 各來源依類別分別以 80%／20% 劃分訓練與測試，再合併對應子集 |

自行蒐集資料以 Labelme 進行標註。資料存在類別不平衡，且部分類別只出現在自行蒐集資料中；研究未使用明確的類別重加權策略。

### 訓練策略

| 設定 | Fine-tuning | 兩階段 Transfer Learning |
| --- | --- | --- |
| 初始化 | 預訓練 YOLOv13n | 預訓練模型 |
| 訓練 Epochs | 300 | Stage 1：100；Stage 2：100 |
| Batch size | 24 | 24 |
| 影像尺寸 | 640 | 640 |
| 凍結層數 | 0 | Stage 1：9；Stage 2：0 |
| Optimizer | auto | AdamW |
| 初始學習率（報告表格） | 0.01 | Stage 1：0.01；Stage 2：0.001 |

兩階段策略先凍結前九層訓練較高階特徵，再解除凍結並降低學習率進行完整微調。報告也記錄了 HSV 色彩變化、平移、縮放、旋轉、Mosaic 等資料增強設定。

### 結果與觀察

- Fine-tuning 與 Transfer Learning 的訓練損失皆呈現收斂趨勢。
- 訓練樣本較多的號誌類別通常有較好的偵測表現，少樣本類別則較具挑戰性。
- 報告指出兩階段策略在驗證結果及訓練效率上具有優勢，但部分段落對不同 mAP 指標的比較敘述不完全一致；因此此處**不宣稱某一策略在所有 mAP 指標均較佳**。請參閱原始報告中的訓練曲線、混淆矩陣及驗證圖表。

## 參考資料

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy FFT Documentation](https://numpy.org/doc/stable/reference/routines.fft.html)
- [EuroSAT Dataset](https://github.com/phelber/eurosat)
- [Keras — Convolutional Autoencoder for Image Denoising](https://keras.io/examples/vision/autoencoder/)
- [YOLOv13 官方程式庫](https://github.com/iMoonLab/yolov13)
- [YOLOv13 論文](https://arxiv.org/abs/2506.17733)
- [Ultralytics 訓練參數文件](https://docs.ultralytics.com/usage/cfg/)