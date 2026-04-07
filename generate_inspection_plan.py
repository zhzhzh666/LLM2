#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能水处理厂巡检计划生成器 (GPT-5 可运行版)
"""

import os
import json
import math
from openai import OpenAI
from tqdm import tqdm
# ======================
# 初始化 OpenAI
# ======================
client = OpenAI()

# ======================
# 配置
# ======================
INPUT_DIR = "inspection_logs"
OUTPUT_DIR = "inspection_plans"

# ======================
# 地图数据
# ======================
MAP_DATA = {
    "start": {"name": "入口", "x": 1.0, "y": 1.0},
    "end": {"name": "出口", "x": 19.0, "y": 1.0},
    "points": {
        "巡检点1_进水pH": {"x": 2.5, "y": 14.5, "task": 1},
        "巡检点2_沉淀池": {"x": 8.5, "y": 13.5, "task": 1},
        "巡检点3_加药系统": {"x": 4.5, "y": 6.5, "task": 2},
        "巡检点4_澄清池": {"x": 10.5, "y": 10.5, "task": 1},
        "巡检点5_过滤床": {"x": 16.5, "y": 14.5, "task": 1},
        "巡检点6_消毒池": {"x": 14.5, "y": 7.5, "task": 1},
        "巡检点7_清水池": {"x": 18.5, "y": 7.5, "task": 2},
        "巡检点8_送水泵": {"x": 16.5, "y": 3.5, "task": 2},
    }
}

# ======================
# 工具函数
# ======================
def get_all_points():
    pts = {}
    pts["入口"] = {"x": 1.0, "y": 1.0}
    for k, v in MAP_DATA["points"].items():
        pts[k] = v
    pts["出口"] = {"x": 19.0, "y": 1.0}
    return pts


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ======================
# GPT-5 调用（核心修复）
# ======================
def call_llm(prompt):
    try:
        response = client.responses.create(
            model="gpt-5",
            input=prompt,
            temperature=0.1,
            max_output_tokens=800,
            response_format={"type": "json_object"}
        )

        content = response.output[0].content[0].text
        return json.loads(content)

    except Exception as e:
        print("LLM错误:", e)
        return None


# ======================
# Prompt
# ======================
def build_prompt(log):
    return f"""
你是巡检调度系统。

上次巡检记录：
{json.dumps(log, ensure_ascii=False)}

输出JSON：
{{
 "decision_reason": "...",
 "inspection_plan": [
   {{"point_name": "巡检点1_进水pH", "action": "task_1"}}
 ]
}}
"""


# ======================
# 路径生成
# ======================
def build_path(plan, point_dict):
    path = []

    path.append({"name": "入口", "pos": (1, 1)})

    for item in plan:
        name = item["point_name"]
        p = point_dict[name]
        path.append({"name": name, "pos": (p["x"], p["y"])})

    path.append({"name": "出口", "pos": (19, 1)})

    return path


def calc_total_distance(path):
    d = 0
    for i in range(len(path) - 1):
        d += distance(path[i]["pos"], path[i + 1]["pos"])
    return round(d, 2)


# ======================
# 主处理
# ======================
def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        log = json.load(f)

    prompt = build_prompt(log)
    result = call_llm(prompt)

    if not result:
        print("⚠️ 使用默认方案")
        result = {
            "decision_reason": "默认",
            "inspection_plan": [
                {"point_name": k, "action": "task_1"}
                for k in MAP_DATA["points"]
            ]
        }

    pts = get_all_points()
    path = build_path(result["inspection_plan"], pts)

    output = {
        "decision_reason": result["decision_reason"],
        "total_distance": calc_total_distance(path),
        "path": path
    }

    return output


# ======================
# main
# ======================
def main():
    if not os.path.exists(INPUT_DIR):
        print("❌ 没有日志目录")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    for f in tqdm(files):
        out = process_file(os.path.join(INPUT_DIR, f))

        save_path = os.path.join(OUTPUT_DIR, "plan_" + f)
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)

    print("✅ 全部完成")


if __name__ == "__main__":
    main()
