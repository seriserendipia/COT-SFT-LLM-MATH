import os, json, re, ast, csv
from collections import defaultdict

# ==== 路径配置（按需修改）====
ROOT = os.path.expanduser("~/LLaMA-Factory")
DATA_DIR = os.path.join(ROOT, "data")
TEST_JSON = os.path.join(DATA_DIR, "test.json")   # gold
PRED_JSONL = os.path.join(ROOT, "saves/Qwen2.5-Coder-1.5B/lora/predict_my_test/predictions.jsonl")  # pred
REPORT_CSV = os.path.join(ROOT, "saves/eval_report.csv")

# ==== 工具函数 ====
def load_array_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                arr.append(json.loads(line))
    return arr

def parse_list_string(s):
    """把 "['Final Answer: 72']" 解析成 'Final Answer: 72'；失败则原样返回字符串。"""
    if s is None:
        return ""
    s = str(s)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list) and v:
            return str(v[0])
    except Exception:
        pass
    return s

def clean_final_prefix(s):
    """去掉 'Final Answer:' / 'Answer:' 前缀"""
    return re.sub(r'^\s*(Final\s*Answer|Answer)\s*:\s*', '', str(s), flags=re.I).strip()

def latex_cleanup(s):
    s = str(s).replace("\\boxed", "")
    s = s.replace("\\,", "").replace(",", "")     # 去千分位
    return s

def to_number_str(s):
    """
    抓“最后一个数字”（含负号/小数/整数）。若想支持分数，可自行扩展。
    返回字符串形式，便于稳定比较（'72' vs '72.0' 可再做标准化）。
    """
    s = latex_cleanup(s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None

def normalize_num_str(ns):
    """把 '72.0' 标准化为 '72'；去掉前导零等细节，统一比较口径。"""
    if ns is None:
        return None
    if "." in ns:
        try:
            v = float(ns)
            if abs(v - int(v)) < 1e-9:
                return str(int(round(v)))
            return str(v).rstrip("0").rstrip(".")
        except Exception:
            return ns.strip()
    # 纯整数
    return ns.lstrip("0") or "0"

def get_pred_text(obj):
    """兼容不同预测字段"""
    for k in ("prediction", "text", "response", "output", "generated_text"):
        if k in obj:
            return str(obj[k])
    if "predictions" in obj and obj["predictions"]:
        return str(obj["predictions"][0])
    # 兜底
    return str(obj)

def strip_prompt_prefix(instruction, prediction):
    """
    如果 prediction 以题干开头，切掉这段前缀，以免把题干里的数字误判为答案。
    做宽松匹配：忽略两端空白与换行差异。
    """
    i = (instruction or "").strip()
    p = (prediction or "").strip()
    # 简单 startswith；需要更鲁棒可以再做模糊比对
    return p[len(i):].lstrip() if i and p.startswith(i) else p

# ==== 读取 gold ====
if not os.path.exists(TEST_JSON):
    raise FileNotFoundError(f"未找到 gold 测试集：{TEST_JSON}")

gold_rows = load_array_json(TEST_JSON)
gold = []
for ex in gold_rows:
    instr = (ex.get("instruction") or "").strip()
    fa0 = parse_list_string(ex.get("final_answer"))
    fa  = clean_final_prefix(fa0)
    gnum = normalize_num_str(to_number_str(fa))
    gold.append({
        "instruction": instr,
        "gold_text": fa,
        "gold_num": gnum,
    })

# ==== 读取 pred ====
if not os.path.exists(PRED_JSONL):
    raise FileNotFoundError(f"未找到预测文件：{PRED_JSONL}")

pred_rows = load_jsonl(PRED_JSONL)
# 顺序列表（优先按顺序对齐）
pred_seq = [( (r.get("instruction") or "").strip(), get_pred_text(r) ) for r in pred_rows]
# also build a map for fallback
pred_map = defaultdict(list)
for ins, txt in pred_seq:
    pred_map[ins].append(txt)

# ==== 对齐 + 评测 ====
n = len(gold)
em = 0
cov_pred = 0
cov_gold = sum(1 for g in gold if g["gold_num"] is not None)

rows = []
mismatches_preview = []
for i, g in enumerate(gold):
    instr = g["instruction"]
    gold_num = g["gold_num"]

    # 先按顺序拿预测；顺序不足时按 instruction 匹配弹出一条
    if i < len(pred_seq):
        pred_text = pred_seq[i][1]
        # 顺序对不上且 instruction 不同，则尝试按 instruction 匹配
        if pred_seq[i][0] != instr and pred_map[instr]:
            pred_text = pred_map[instr].pop(0)
    else:
        pred_text = pred_map[instr].pop(0) if pred_map[instr] else ""

    # 去掉题干前缀再抽数字
    pred_cut = strip_prompt_prefix(instr, pred_text)
    pnum = normalize_num_str(to_number_str(pred_cut))

    if pnum is not None:
        cov_pred += 1

    correct = (gold_num is not None and pnum == gold_num)
    em += int(correct)

    if not correct and len(mismatches_preview) < 10:
        preview = pred_cut[:150].replace("\n", " ")
        if len(pred_cut) > 150:
            preview += "..."
        mismatches_preview.append((i, gold_num, pnum, preview))

    rows.append({
        "idx": i,
        "instruction": instr,
        "gold_text": g["gold_text"],
        "gold_num": gold_num,
        "prediction": pred_text,
        "pred_num": pnum,
        "correct": int(correct),
    })

# ==== 汇总 ====
print(f"Samples compared: {n}")
print(f"Gold with extractable number: {cov_gold}")
print(f"Predictions with extractable number: {cov_pred}")
acc = (em / n * 100.0) if n else 0.0
print(f"Exact-Match (numeric): {em}/{n} = {acc:.2f}%")

print("\nMismatches (up to 10):")
for idx, gnum, pnum, raw in mismatches_preview:
    print(f"[{idx}] gold={gnum} pred={pnum} | pred_raw='{raw}'")

# ==== 导出逐题报告 ====
os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
with open(REPORT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n[OK] 详细结果已保存: {REPORT_CSV}")