import os

# 入力ファイルと出力ファイルのパスを設定
input_file_path = "Data/W0401/palate_aligned/1.05_703_palate_xyz.data"
output_file_path = "Data/W0401/palate_aligned/1.05_703_palate.data"

# 出力先のディレクトリが存在しない場合は自動作成
output_dir = os.path.dirname(output_file_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

try:
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # 空行はスキップ
            if not line.strip():
                continue
            
            # 半角スペースやタブなどの空白文字で列を分割
            columns = line.split()
            
            # データの安全性を確認（少なくとも3列以上ある行を処理）
            if len(columns) >= 3:
                # Pythonのインデックスは0から始まるため、2列目は[1]、3列目は[2]となります
                col2 = columns[1]
                col3 = columns[2]
                
                # 新しいファイルに2列目と3列目をスペース区切りで書き出し
                outfile.write(f"{col2} {col3}\n")
                
    print(f"抽出が完了しました！\n保存先: {output_file_path}")

except FileNotFoundError:
    print(f"エラー: 元ファイルが見つかりません。パスが正しいかご確認ください。\n確認したパス: {input_file_path}")
