from flask import Flask, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# 設定檔案儲存路徑，指向你 Angular 專案的 public 資料夾
SAVE_DIR = "/home/t1204-3060/HOWARD/howard/drone-dashboard/public/data"

@app.route('/upload', methods=['POST'])
def upload_file():
    # 接收樹莓派傳來的 BMP 或 TXT
    file = request.files['file']
    file_type = request.form.get('type') # bmp 或 txt
    
    if file_type == 'bmp':
        save_path = os.path.join(SAVE_DIR, 'bmp', 'latest.bmp')
    else:
        save_path = os.path.join(SAVE_DIR, 'txt', 'latest.txt')
    
    file.save(save_path)
    
    # 【這裡跑辨識 Code】
    # result = my_cnn_model.predict(save_path)
    
    return {"status": "success", "message": f"{file_type} uploaded"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)