# CPU 人臉辨識

使用 OpenCV Zoo 的 YuNet 偵測人臉、SFace 對齊並擷取特徵，再用餘弦相似度比對本機已註冊的人臉。模型下載完成後，全程可離線執行。支援照片註冊、照片辨識與單人攝影機辨識。

## 安裝

從專案根目錄建立並啟用獨立的 Conda 環境，再把依賴安裝到該環境。

```sh
conda create -n face-recognition-cpu python=3.13 pip
conda activate face-recognition-cpu
python -m pip install -r requirements.txt
```

建立 `models/` 資料夾，從 OpenCV Zoo 的檔案頁下載下列兩個 ONNX 檔。使用頁面上的下載按鈕取得實際模型檔。

| 模型檔 | 上游授權 |
| --- | --- |
| [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | [MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE) |
| [face_recognition_sface_2021dec.onnx](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) | [Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE) |

## 使用方式

將經本人同意使用的照片放在本機的 `local-data/`。請在已啟用的專案環境中執行下列指令。

```sh
python face_recognition.py enroll person01 local-data/reference.jpg
python face_recognition.py identify local-data/probe.jpg
python face_recognition.py live --camera 0 --stable-frames 5
```

- 註冊及照片辨識各需偵測到一張臉。標籤可用英文字母、數字、連字號與底線，須以英數開頭，長度最多 40 字元。同名標籤會更新本機向量。刪除 `local-data/sface-v1/id-person01.npy` 即可移除該筆註冊。
- 攝影機模式需要桌面視窗及攝影機權限。按 `q` 或 `Esc` 結束。連續達到指定幀數才顯示身分，遇到未匹配、無臉或多張臉時重新計數。
- `identify` 和 `live` 可用 `--threshold` 調整餘弦相似度門檻。預設 `0.363` 取自 OpenCV 教學的 LFW 評估。請依實際光線、攝影機與測試結果調整。此展示未實作活體偵測。分數是相似度，無法直接當作身分正確率。 [OpenCV 人臉辨識教學](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- 可在子指令前以 `--models` 和 `--database` 指定本機目錄。若曾使用其他人臉模型，請用 SFace 重新註冊。不同模型產生的向量與分數門檻各自適用於原本的模型。

## 發布與資料

預設工作目錄中的模型、照片和人臉向量均由 Git 忽略。程式只將註冊向量與標籤儲存在本機，攝影機畫面不寫入磁碟。請將自訂的資料目錄放在 Git 倉庫外，並避免讓照片、影片、特徵向量或可辨識身分的標籤進入 Git 紀錄。公開倉庫可只提交本程式、README、依賴清單和 Git 忽略設定。上表授權適用於各自的上游模型。本專案的程式授權由發佈者確認權利後另行指定。
