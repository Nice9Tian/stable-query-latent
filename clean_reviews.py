import os
import json
import pandas as pd


BASE_DIR = r"C:\Users\admin\Desktop\database"
REVIEWS_DIR = r"C:\Users\admin\Desktop\database\Steam Games Metadata and Player Reviews (2020–2024\Game Reviews"
GAMES_JSON  = r"C:\Users\admin\Desktop\database\Steam Games Metadata and Player Reviews (2020–2024\games.json"
OUTPUT_BASE = os.path.join(BASE_DIR, "game_review_cleaned")


def get_output_dir(base: str) -> str:
    """若 base 已存在则依次尝试 base_1, base_2, ... 直到找到不重复的路径。"""
    if not os.path.exists(base):
        return base
    num = 1
    while True:
        candidate = f"{base}_{num}"
        if not os.path.exists(candidate):
            return candidate
        num += 1


def find_latest_cleaned_dir(base: str) -> str | None:
    """找到 base_num 中 num 最大的已存在文件夹；若连 base 本身都不存在则返回 None。"""
    latest = None
    latest_num = -1

    if os.path.exists(base):
        latest = base
        latest_num = 0

    num = 1
    while True:
        candidate = f"{base}_{num}"
        if os.path.exists(candidate):
            latest = candidate
            latest_num = num
            num += 1
        else:
            break

    return latest


def filter_by_review_length(src_dir: str, output_dir: str, n: int):
    """保留 review 字段长度 >= n 的行；评论数为 0 的文件不复制。"""
    csv_files = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
    print(f"\n[按长度清洗] 源目录：{src_dir}")
    print(f"输出目录：{output_dir}  |  最小 review 长度：{n}")
    print(f"共 {len(csv_files)} 个文件\n")

    skipped = 0
    for filename in csv_files:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        try:
            df = pd.read_csv(src_path)
        except Exception as e:
            print(f"  [跳过] {filename} 读取失败：{e}")
            skipped += 1
            continue

        if "review" not in df.columns:
            print(f"  [跳过] {filename} 缺少 review 列")
            skipped += 1
            continue

        before = len(df)
        df = df[df["review"].fillna("").str.len() >= n]
        after = len(df)

        if after == 0:
            print(f"  [丢弃] {filename}：过滤后无剩余行，不复制")
            skipped += 1
            continue

        df.to_csv(dst_path, index=False)
        print(f"  {filename}：{before} → {after} 行")

    print(f"\n完成！保存至：{output_dir}（跳过/丢弃 {skipped} 个文件）")


def filter_by_review_count(src_dir: str, output_dir: str, min_count: int):
    """只复制评论数（行数）>= min_count 的 CSV 文件，不满足的整个文件不复制。"""
    csv_files = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
    print(f"\n[按评论数筛选] 源目录：{src_dir}")
    print(f"输出目录：{output_dir}  |  最小评论数：{min_count}")
    print(f"共 {len(csv_files)} 个文件\n")

    kept = 0
    dropped = 0
    for filename in csv_files:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        try:
            df = pd.read_csv(src_path)
        except Exception as e:
            print(f"  [跳过] {filename} 读取失败：{e}")
            dropped += 1
            continue

        count = len(df)
        if count < min_count:
            print(f"  [丢弃] {filename}：{count} 条评论 < {min_count}，不复制")
            dropped += 1
        else:
            df.to_csv(dst_path, index=False)
            print(f"  [保留] {filename}：{count} 条评论")
            kept += 1

    print(f"\n完成！保留 {kept} 个文件，丢弃 {dropped} 个文件。保存至：{output_dir}")


def load_games_meta() -> dict:
    """加载 games.json，返回以 game_id 字符串为键的字典。"""
    with open(GAMES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def combine_reviews(src_dir: str, output_dir: str, with_meta: bool = False):
    """
    每个 CSV 输出一个同名 .json 文件，内容为 review 列表。
    with_meta=True 时，在列表最前面插入三个元数据字符串：
      [0] detailed_description, [1] about_the_game, [2] short_description
    文件名格式须为 gameid_采集id.csv，game_id 从中解析。
    """
    all_files = [
        f for f in os.listdir(src_dir)
        if f.endswith(".csv") or f.endswith(".json")
    ]
    print(f"\n[合并 review] 源目录：{src_dir}")
    print(f"输出目录：{output_dir}  |  附加游戏元数据：{'是' if with_meta else '否'}")
    print(f"共 {len(all_files)} 个文件（CSV + JSON）\n")

    games_meta = load_games_meta() if with_meta else {}
    skipped = 0

    for filename in all_files:
        src_path = os.path.join(src_dir, filename)

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(src_path)
                if "review" not in df.columns:
                    print(f"  [跳过] {filename} 缺少 review 列")
                    skipped += 1
                    continue
                reviews = df["review"].fillna("").astype(str).tolist()
            else:
                with open(src_path, "r", encoding="utf-8") as f:
                    reviews = json.load(f)
                if not isinstance(reviews, list):
                    print(f"  [跳过] {filename} JSON 格式不是数组")
                    skipped += 1
                    continue
        except Exception as e:
            print(f"  [跳过] {filename} 读取失败：{e}")
            skipped += 1
            continue

        stem = filename.rsplit(".", 1)[0]  # 去掉扩展名
        reviews = [str(r) for r in reviews]

        if with_meta:
            game_id = stem.split("_")[0]
            meta = games_meta.get(game_id, {})
            prefix = [
                meta.get("detailed_description", ""),
                meta.get("about_the_game", ""),
                meta.get("short_description", ""),
            ]
            if not any(prefix):
                print(f"  [警告] {filename}：games.json 中未找到 game_id={game_id}，元数据为空")
            data = prefix + reviews
        else:
            data = reviews

        dst_path = os.path.join(output_dir, stem + ".json")
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {filename}：{len(reviews)} 条 review → {os.path.basename(dst_path)}")

    print(f"\n完成！保存至：{output_dir}（跳过 {skipped} 个文件）")


def ask_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("  请输入有效整数。")


def ask_mode() -> str:
    print("\n请选择操作模式：")
    print("  1. 按 review 内容长度过滤（保留长度 >= n 的行）")
    print("  2. 按评论数筛选（评论数不足的文件整体丢弃）")
    print("  3. 两步连续清洗（先按评论数，再按长度）")
    print("  4. 合并 review（每文件输出 JSON 数组）")
    print("  5. 合并 review + 附加游戏元数据（detailed_description / about_the_game / short_description 置于数组前三位）")
    while True:
        choice = input("请输入 1 / 2 / 3 / 4 / 5：").strip()
        if choice in ("1", "2", "3", "4", "5"):
            return choice
        print("  无效输入，请输入 1～5。")


if __name__ == "__main__":
    # ── 1. 确定输入源目录 ──────────────────────────────────────────────
    print("请输入源文件夹路径（直接回车则自动寻找 num 最大的 game_review_cleaned_num）：")
    user_input = input("路径：").strip()

    if user_input:
        src_dir = user_input
        if not os.path.exists(src_dir):
            print(f"路径不存在：{src_dir}")
            exit(1)
    else:
        src_dir = find_latest_cleaned_dir(OUTPUT_BASE)
        if src_dir is None:
            print("未找到任何 game_review_cleaned* 文件夹，将使用原始 Game Reviews 目录。")
            src_dir = REVIEWS_DIR
        else:
            print(f"自动选择源目录：{src_dir}")

    # ── 2. 选择模式 ────────────────────────────────────────────────────
    mode = ask_mode()

    if mode in ("4", "5"):
        output_dir = get_output_dir(OUTPUT_BASE)
        os.makedirs(output_dir)
        print(f"新输出目录：{output_dir}")
        combine_reviews(src_dir, output_dir, with_meta=(mode == "5"))
    else:
        # ── 3. 确定输出目录 ────────────────────────────────────────────
        output_dir = get_output_dir(OUTPUT_BASE)
        os.makedirs(output_dir)
        print(f"新输出目录：{output_dir}")

        if mode == "1":
            n = ask_int("请输入 review 最小长度 n：")
            filter_by_review_length(src_dir, output_dir, n)

        elif mode == "2":
            min_count = ask_int("请输入最小评论数：")
            filter_by_review_count(src_dir, output_dir, min_count)

        elif mode == "3":
            min_count = ask_int("请输入最小评论数（第一步）：")
            n = ask_int("请输入 review 最小长度 n（第二步）：")
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                filter_by_review_count(src_dir, tmp, min_count)
                filter_by_review_length(tmp, output_dir, n)
