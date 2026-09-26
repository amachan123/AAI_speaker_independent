#口蓋データにはx,y,z,θ1,θ2,errorが含まれている。
#そこからx,y,zのみを抽出する。

import os

# 入力ファイルと出力ファイルのパスを設定
input_file_path = "Data/W0401/palate/lpf_1.05_hfS20200220_D2020022010703mov_0_bar_tip_POS_angle.data"
output_file_path = "Data/W0401/palate/1.05_703_palate_xyz.data"

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
                # インデックス0(1列目)、1(2列目)、2(3列目)を抽出
                col1 = columns[0]
                col2 = columns[1]
                col3 = columns[2]
                
                # 新しいファイルに1〜3列目をスペース区切りで書き出し
                outfile.write(f"{col1} {col2} {col3}\n")
                
    print(f"抽出が完了しました！\n保存先: {output_file_path}")

except FileNotFoundError:
    print(f"エラー: 元ファイルが見つかりません。パスが正しいかご確認ください。\n確認したパス: {input_file_path}")
