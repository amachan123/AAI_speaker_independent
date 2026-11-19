from pathlib import Path
import re

# =========================================================
# 設定
# =========================================================
INPUT_DIR = Path("Data/Onso_row/kana")
OUTPUT_DIR = Path("Data/Onso_row/phoneme")

SPEAKERS = [
    "W0301",
    "W0501",
    "W0701",
    "W0801",
]


# =========================================================
# ひらがな → Julius音素列
# =========================================================
KANA_TO_PHONE = {
    # 母音
    "あ": ["a"],
    "い": ["i"],
    "う": ["u"],
    "え": ["e"],
    "お": ["o"],

    # k
    "か": ["k", "a"],
    "き": ["k", "i"],
    "く": ["k", "u"],
    "け": ["k", "e"],
    "こ": ["k", "o"],
    "きゃ": ["ky", "a"],
    "きゅ": ["ky", "u"],
    "きょ": ["ky", "o"],

    # g
    "が": ["g", "a"],
    "ぎ": ["g", "i"],
    "ぐ": ["g", "u"],
    "げ": ["g", "e"],
    "ご": ["g", "o"],
    "ぎゃ": ["gy", "a"],
    "ぎゅ": ["gy", "u"],
    "ぎょ": ["gy", "o"],

    # s / sh
    "さ": ["s", "a"],
    "し": ["sh", "i"],
    "す": ["s", "u"],
    "せ": ["s", "e"],
    "そ": ["s", "o"],
    "しゃ": ["sh", "a"],
    "しゅ": ["sh", "u"],
    "しょ": ["sh", "o"],

    # z / j
    "ざ": ["z", "a"],
    "じ": ["j", "i"],
    "ず": ["z", "u"],
    "ぜ": ["z", "e"],
    "ぞ": ["z", "o"],
    "じゃ": ["j", "a"],
    "じゅ": ["j", "u"],
    "じょ": ["j", "o"],
    "じぇ": ["j", "e"],

    # t / ch / ts
    "た": ["t", "a"],
    "ち": ["ch", "i"],
    "つ": ["ts", "u"],
    "て": ["t", "e"],
    "と": ["t", "o"],
    "ちゃ": ["ch", "a"],
    "ちゅ": ["ch", "u"],
    "ちょ": ["ch", "o"],
    "ちぇ": ["ch", "e"],
    "てぃ": ["t", "i"],

    # d
    "だ": ["d", "a"],
    "ぢ": ["j", "i"],
    "づ": ["z", "u"],
    "で": ["d", "e"],
    "ど": ["d", "o"],
    "でぃ": ["d", "i"],
    "でゅ": ["dy", "u"],

    # n
    "な": ["n", "a"],
    "に": ["n", "i"],
    "ぬ": ["n", "u"],
    "ね": ["n", "e"],
    "の": ["n", "o"],
    "にゃ": ["ny", "a"],
    "にゅ": ["ny", "u"],
    "にょ": ["ny", "o"],

    # h / f
    "は": ["h", "a"],
    "ひ": ["h", "i"],
    "ふ": ["f", "u"],
    "へ": ["h", "e"],
    "ほ": ["h", "o"],
    "ひゃ": ["hy", "a"],
    "ひゅ": ["hy", "u"],
    "ひょ": ["hy", "o"],
    "ふぁ": ["f", "a"],
    "ふぃ": ["f", "i"],
    "ふぇ": ["f", "e"],
    "ふぉ": ["f", "o"],

    # b
    "ば": ["b", "a"],
    "び": ["b", "i"],
    "ぶ": ["b", "u"],
    "べ": ["b", "e"],
    "ぼ": ["b", "o"],
    "びゃ": ["by", "a"],
    "びゅ": ["by", "u"],
    "びょ": ["by", "o"],

    # p
    "ぱ": ["p", "a"],
    "ぴ": ["p", "i"],
    "ぷ": ["p", "u"],
    "ぺ": ["p", "e"],
    "ぽ": ["p", "o"],
    "ぴゃ": ["py", "a"],
    "ぴゅ": ["py", "u"],
    "ぴょ": ["py", "o"],

    # m
    "ま": ["m", "a"],
    "み": ["m", "i"],
    "む": ["m", "u"],
    "め": ["m", "e"],
    "も": ["m", "o"],
    "みゃ": ["my", "a"],
    "みゅ": ["my", "u"],
    "みょ": ["my", "o"],

    # y
    "や": ["y", "a"],
    "ゆ": ["y", "u"],
    "よ": ["y", "o"],

    # r
    "ら": ["r", "a"],
    "り": ["r", "i"],
    "る": ["r", "u"],
    "れ": ["r", "e"],
    "ろ": ["r", "o"],
    "りゃ": ["ry", "a"],
    "りゅ": ["ry", "u"],
    "りょ": ["ry", "o"],

    # w
    "わ": ["w", "a"],
    "うぃ": ["w", "i"],
    "うぇ": ["w", "e"],
    "うぉ": ["w", "o"],

    # 特殊
    "ん": ["N"],
    "っ": ["q"],
}


VOWELS = {"a", "i", "u", "e", "o"}


# =========================================================
# 1文を音素列へ変換
# =========================================================
def kana_to_phones(text):

    phones = []
    i = 0

    while i < len(text):

        # -----------------------------
        # 読点 → sp
        # -----------------------------
        if text[i] == "、":
            phones.append("sp")
            i += 1
            continue

        # -----------------------------
        # 長音記号
        # -----------------------------
        if text[i] == "ー":

            if not phones:
                raise ValueError(
                    "文頭に長音記号があります"
                )

            if phones[-1] in VOWELS:
                phones[-1] += ":"
            else:
                raise ValueError(
                    f"長音記号の直前が母音ではありません: "
                    f"{phones[-1]}"
                )

            i += 1
            continue

        # -----------------------------
        # 2文字仮名を優先
        # -----------------------------
        if i + 1 < len(text):

            kana2 = text[i:i + 2]

            if kana2 in KANA_TO_PHONE:
                phones.extend(
                    KANA_TO_PHONE[kana2]
                )
                i += 2
                continue

        # -----------------------------
        # 1文字仮名
        # -----------------------------
        kana1 = text[i]

        if kana1 in KANA_TO_PHONE:
            phones.extend(
                KANA_TO_PHONE[kana1]
            )
            i += 1
            continue

        # 空白
        if kana1.isspace():
            i += 1
            continue

        raise ValueError(
            f"未登録文字: {kana1}"
        )

    return phones


# =========================================================
# 1話者分を変換
# =========================================================
def convert_speaker(speaker):

    input_path = (
        INPUT_DIR
        / f"{speaker}.txt"
    )

    output_path = (
        OUTPUT_DIR
        / f"{speaker}.txt"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"入力ファイルなし: {input_path}"
        )

    lines = input_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    output_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ---------------------------------
        # 001 文章
        # ---------------------------------
        match = re.match(
            r"^(\d{3})\s+(.+)$",
            line
        )

        if not match:
            raise ValueError(
                f"{speaker}: "
                f"行形式が不正です: {line}"
            )

        utt_id = match.group(1)
        kana = match.group(2)

        try:
            phones = kana_to_phones(
                kana
            )

        except Exception as exc:
            raise ValueError(
                f"{speaker}/{utt_id}: "
                f"{exc}"
            ) from exc

        output_lines.append(
            f"{utt_id} "
            f"{' '.join(phones)}"
        )

    output_path.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8"
    )

    print(
        f"{speaker}: "
        f"{len(output_lines)} utterances"
    )

    print(
        f"Saved: {output_path}"
    )


# =========================================================
# main
# =========================================================
def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for speaker in SPEAKERS:

        convert_speaker(
            speaker
        )


if __name__ == "__main__":
    main()
