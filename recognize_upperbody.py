import cv2 as cv
import os

# 現在のフォルダ
current_dir = os.path.dirname(os.path.abspath(__file__))

# 上半身用カスケードファイルのパス
model_path = os.path.join(current_dir, "models/haarcascade_upperbody.xml")

# カスケード分類器の読み込み
body_cascade = cv.CascadeClassifier(model_path)

# カメラからの入力を開始
cap = cv.VideoCapture(0)

if not body_cascade.load(model_path):
    print("カスケードファイルが読み込めませんでした")
    exit()

while True:
    # フレームを読み込む
    ret, frame = cap.read()

    if not ret:
        continue

    # グレースケールに変換
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 上半身を検出
    bodies = body_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(100, 100))

    # 検出した上半身を処理
    for (x, y, w, h) in bodies:
        RATE = 1.1  # 上半身のサイズを少し拡大
        enlarged_w = int(w * RATE)
        enlarged_h = int(h * RATE)

        x = max(0, x - (enlarged_w - w) // 2)
        y = max(0, y - (enlarged_h - h) // 2)

        end_x = min(x + enlarged_w, frame.shape[1])
        end_y = min(y + enlarged_h, frame.shape[0])
        start_x = max(0, end_x - enlarged_w)
        start_y = max(0, end_y - enlarged_h)

        # 上半身の周りに矩形を描画
        cv.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    # 結果を表示
    cv.imshow('Upper Body Detection', frame)

    # escキーで終了
    if cv.waitKey(20) & 0xFF == 27:
        break

# リソースを解放
cap.release()
cv.destroyAllWindows()
