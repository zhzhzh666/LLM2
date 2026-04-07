#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能仓储巡检计划生成器（中文出入口支持）
- 起点: "入口"
- 终点: "出口"
- 中间点由 Qwen3-Max 决策 + 最短路径排序
"""

import os
import json
import yaml
import re
import math
import dashscope
from dashscope import Generation

# ======================
# 配置文件路径
# ======================
MAP_YAML = "map.yaml"
PATROL_POINTS_YAML = "patrol_points_arm.yaml"
INSPECTION_LOG_JSON = "inspection_log_20260213.json"
OUTPUT_PLAN_JSON = "today_inspection_plan.json"

# 初始化 DashScope API
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


# ======================
# 工具函数
# ======================

def load_map_metadata():
    """加载地图元数据（用于验证，路径规划使用世界坐标）"""
    with open(MAP_YAML, 'r', encoding='utf-8') as f:
        meta = yaml.safe_load(f)
    origin = meta.get("origin", [0, 0, 0])
    return {
        "resolution": float(meta["resolution"]),
        "origin_x": float(origin[0]),
        "origin_y": float(origin[1])
    }


def euclidean_distance(p1, p2):
    """计算两点间欧氏距离（单位：米）"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def solve_tsp_with_fixed_ends(middle_waypoints, point_dict):
    """
    生成完整路径: "入口" → [中间任务点] → "出口"
    """
    # === 获取起点 "入口" ===
    if "入口" not in point_dict:
        raise ValueError("❌ patrol_points_arm.yaml 中缺少 '入口' 点")
    ent = point_dict["入口"]
    start_point = {
        "point_name": "入口",
        "position": [round(ent["x"], 3), round(ent["y"], 3), round(ent["z"], 3)],
        "action": "arrive",
        "arm_task_type": 0
    }

    # === 获取终点 "出口" ===
    if "出口" not in point_dict:
        raise ValueError("❌ patrol_points_arm.yaml 中缺少 '出口' 点")
    ex = point_dict["出口"]
    end_point = {
        "point_name": "出口",
        "position": [round(ex["x"], 3), round(ex["y"], 3), round(ex["z"], 3)],
        "action": "arrive",
        "arm_task_type": 0
    }

    # === 若无中间点，直接返回 [入口, 出口] ===
    if not middle_waypoints:
        return [start_point, end_point]

    # === 贪心排序中间点：从 "入口" 开始，依次选最近 ===
    ordered = []
    unvisited = middle_waypoints[:]
    current_pos = start_point["position"][:2]

    while unvisited:
        nearest = min(unvisited, key=lambda w: euclidean_distance(current_pos, w["position"][:2]))
        ordered.append(nearest)
        current_pos = nearest["position"][:2]
        unvisited.remove(nearest)

    return [start_point] + ordered + [end_point]


def extract_json_from_text(text: str):
    """从 LLM 输出中提取合法 JSON"""
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            json.loads(match.group(1))
            return match.group(1)
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return match.group(0)
        except json.JSONDecodeError:
            pass

    return None


# ======================
# 数据加载
# ======================

def load_data():
    """加载巡检点和历史日志"""
    with open(PATROL_POINTS_YAML, "r", encoding="utf-8") as f:
        points = yaml.safe_load(f)["patrol_points_arm"]
    point_dict = {p["name"]: p for p in points}

    with open(INSPECTION_LOG_JSON, "r", encoding="utf-8") as f:
        log = json.load(f)

    return point_dict, log


# ======================
# LLM 交互
# ======================

def build_prompt(point_dict, last_log):
    """构造提示词（排除入口/出口）"""
    # 只保留任务点（排除 "入口" 和 "出口"）
    task_points = {
        name: p for name, p in point_dict.items()
        if name not in ["入口", "出口"]
    }

    # 地图信息
    map_lines = []
    for name, p in task_points.items():
        task_desc = {0: "无任务", 1: "任务1（扫描）", 2: "任务2（抓取）"}.get(p["arm_task"], "未知")
        map_lines.append(f"- {name}: ({p['x']:.2f}, {p['y']:.2f}) → {task_desc}")
    map_info = "\n".join(map_lines) or "无任务点"

    # 历史日志
    log_lines = []
    for pt in last_log.get("visited_points", []):
        status = last_log["tasks_executed"].get(pt, "未记录")
        log_lines.append(f"- {pt}: {status}")
    log_info = "\n".join(log_lines) or "无历史记录"

    anomalies = last_log.get("anomalies", [])
    if anomalies:
        log_info += "\n异常记录:\n" + "\n".join(f"  • {a}" for a in anomalies)

    prompt = f"""你是一个智能仓储巡检调度系统。请根据以下信息生成本次巡检计划。

【地图信息】（仅任务点）
{map_info}

【历史巡检日志】
{log_info}

【决策规则】
1. 必须重试上次失败的点（特别是状态含 'failed' 的）。
2. 上次成功的点可跳过。
3. 不要选择 '入口' 或 '出口'（它们仅用于路径起终点）。
4. 输出必须是严格 JSON，包含：
   - "decision_reason": 字符串（简要说明）
   - "inspection_plan": 数组，每项含 "point_name" 和 "action"
     - action 取值: "arrive", "task_1", "task_2"

【重要】
- 只输出 JSON 内容，不要任何解释、注释、Markdown 或额外文字。
- 不要包含 ```json 或 ```

- 确保输出可被 Python json.loads() 直接解析。

示例输出：
{{"decision_reason": "货架B区上次抓取失败，需重试。","inspection_plan": [{{"point_name": "货架B区", "action": "task_2"}}]}}
"""
    return prompt


def call_llm(prompt):
    """调用 Qwen3-Max API"""
    try:
        response = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen3-max",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=0.0,
            timeout=15.0
        )
        if response.status_code != 200:
            print(f"❌ Qwen API 错误: {response.code} - {response.message}")
            return None

        output_text = response.output.choices[0].message.content
        json_str = extract_json_from_text(output_text)
        return json.loads(json_str) if json_str else None
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        return None


def enrich_with_coordinates(llm_plan, point_dict):
    """绑定真实坐标，并过滤出入口"""
    enriched = []
    action_map = {
        "arrive": "arrive",
        "task_1": "execute_arm_task_1",
        "task_2": "execute_arm_task_2"
    }

    for item in llm_plan.get("inspection_plan", []):
        name = item.get("point_name")
        if not name or name not in point_dict:
            continue
        # 跳过出入口（LLM 不应选，但防御性处理）
        if name in ["入口", "出口"]:
            continue

        p = point_dict[name]
        enriched.append({
            "point_name": name,
            "position": [round(p["x"], 3), round(p["y"], 3), round(p["z"], 3)],
            "action": action_map.get(item.get("action"), "unknown"),
            "arm_task_type": p["arm_task"]
        })
    return enriched


# ======================
# 主程序
# ======================

def main():
    print("🔍 正在加载地图与巡检数据...")
    try:
        _ = load_map_metadata()  # 验证地图存在
        point_dict, last_log = load_data()
    except Exception as e:
        print(f"🛑 初始化失败: {e}")
        return

    print("🧠 正在调用 Qwen3-Max 生成巡检决策...")
    prompt = build_prompt(point_dict, last_log)
    llm_output = call_llm(prompt)

    middle_waypoints = []
    decision_reason = "无任务点，仅通行"
    if llm_output:
        decision_reason = llm_output.get("decision_reason", "无说明")
        middle_waypoints = enrich_with_coordinates(llm_output, point_dict)
        print(f"✅ 决策理由: {decision_reason}")
    else:
        print("⚠️ 无法获取 LLM 决策，仅生成通行路径")

    print("🛣️ 正在规划固定起终点的最短路径...")
    try:
        ordered_waypoints = solve_tsp_with_fixed_ends(middle_waypoints, point_dict)
    except ValueError as e:
        print(e)
        return

    # 计算总路径长度（米）
    total_dist = 0.0
    for i in range(1, len(ordered_waypoints)):
        p1 = ordered_waypoints[i-1]["position"][:2]
        p2 = ordered_waypoints[i]["position"][:2]
        total_dist += euclidean_distance(p1, p2)

    # 构建最终输出
    final_plan = {
        "timestamp": "2026-02-14T10:00:00",
        "source": "Qwen3Max_TSP_EntranceToExit_Chinese",
        "decision_reason": decision_reason,
        "total_distance_meters": round(total_dist, 2),
        "waypoints": ordered_waypoints
    }

    # 保存到文件
    with open(OUTPUT_PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(final_plan, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 巡检计划已生成 → {OUTPUT_PLAN_JSON}")
    for i, wp in enumerate(ordered_waypoints, 1):
        pos = wp["position"]
        print(f"  {i}. {wp['point_name']} @ ({pos[0]}, {pos[1]}) → {wp['action']}")
    print(f"\n📏 总路径长度: {total_dist:.2f} 米")


if __name__ == "__main__":
    main()
