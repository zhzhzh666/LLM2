#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能水处理厂巡检计划生成器 (基于提供的地图)
- 场景：水处理厂 (沉淀池、过滤床、消毒池等)
- 功能：LLM 决策 + 贪心路径规划 + 地图可视化
"""
import os
import json
import yaml
import math
import dashscope
from dashscope import Generation

# 绘图库导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.path import Path
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ 未检测到 matplotlib，将跳过地图可视化。请运行: pip install matplotlib")

# ======================
# 配置与常量
# ======================
PATROL_POINTS_YAML = "patrol_points_arm.yaml"
INSPECTION_LOG_JSON = "inspection_log_20260213.json"
OUTPUT_PLAN_JSON = "today_inspection_plan.json"
OUTPUT_MAP_IMAGE = "water_plant_route_map.png"

# 初始化 API
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ======================
# 1. 地图数据定义 (基于上传的图片)
# ======================
# 定义所有关键点的世界坐标 (单位：米)
# 坐标系假设：左下角为 (0,0)，向右为 X 正，向上为 Y 正
MAP_DATA = {
    "resolution": 0.5,  # 绘图分辨率，仅用于像素转换参考
    
    # 起点与终点
    "start": {"name": "入口", "x": 1.0, "y": 1.0, "type": "start"},
    "end": {"name": "出口", "x": 19.0, "y": 1.0, "type": "end"},
    
    # 巡检任务点 (对应图中的红点编号)
    "points": {
        "巡检点1_进水pH": {"x": 2.5, "y": 14.5, "task": 1, "desc": "进水pH计检测"},
        "巡检点2_沉淀池": {"x": 8.5, "y": 13.5, "task": 1, "desc": "沉淀池液位/状态"},
        "巡检点3_加药系统": {"x": 4.5, "y": 6.5, "task": 2, "desc": "加药泵状态检查"},
        "巡检点4_澄清池": {"x": 10.5, "y": 10.5, "task": 1, "desc": "圆形澄清池取样"},
        "巡检点5_过滤床": {"x": 16.5, "y": 14.5, "task": 1, "desc": "过滤床压差监测"},
        "巡检点6_消毒池": {"x": 14.5, "y": 7.5, "task": 1, "desc": "消毒余氯检测"},
        "巡检点7_清水池": {"x": 18.5, "y": 7.5, "task": 2, "desc": "清水池水位检查"},
        "巡检点8_送水泵": {"x": 16.5, "y": 3.5, "task": 2, "desc": "送水泵振动/温度"},
    },
    
    # 障碍区域 (用于绘图展示，简单矩形表示)
    "obstacles": [
        {"name": "障碍区域B", "x": 2.0, "y": 2.0, "w": 3.0, "h": 2.0}, # 左下设备
        {"name": "障碍区域A", "x": 14.0, "y": 4.0, "w": 6.0, "h": 2.5}, # 送水泵上方
        {"name": "障碍区域C", "x": 18.0, "y": 7.0, "w": 2.0, "h": 2.0}, # 清水池右侧
    ]
}

def get_all_patrol_points():
    """构建完整的巡检点字典，包含起点和终点"""
    points_dict = {}
    
    # 添加起点
    s = MAP_DATA["start"]
    points_dict[s["name"]] = {"name": s["name"], "x": s["x"], "y": s["y"], "z": 0.0, "arm_task": 0}
    
    # 添加任务点
    for name, info in MAP_DATA["points"].items():
        points_dict[name] = {
            "name": name, 
            "x": info["x"], 
            "y": info["y"], 
            "z": 0.0, 
            "arm_task": info["task"],
            "desc": info.get("desc", "")
        }
        
    # 添加终点
    e = MAP_DATA["end"]
    points_dict[e["name"]] = {"name": e["name"], "x": e["x"], "y": e["y"], "z": 0.0, "arm_task": 0}
    
    return points_dict

# ======================
# 2. 核心算法 (保持不变，适配新数据)
# ======================
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def solve_tsp_with_fixed_ends(middle_waypoints, point_dict):
    """贪心算法生成路径：入口 -> 中间点 -> 出口"""
    if "入口" not in point_dict or "出口" not in point_dict:
        raise ValueError("缺少入口或出口定义")
        
    start_node = point_dict["入口"]
    end_node = point_dict["出口"]
    
    start_point = {
        "point_name": "入口",
        "position": [start_node["x"], start_node["y"], start_node["z"]],
        "action": "arrive",
        "arm_task_type": 0
    }
    
    end_point = {
        "point_name": "出口",
        "position": [end_node["x"], end_node["y"], end_node["z"]],
        "action": "arrive",
        "arm_task_type": 0
    }
    
    if not middle_waypoints:
        return [start_point, end_point]
    
    # 贪心排序
    ordered = []
    unvisited = middle_waypoints[:]
    current_pos = start_point["position"][:2]
    
    while unvisited:
        nearest = min(unvisited, key=lambda w: euclidean_distance(current_pos, w["position"][:2]))
        ordered.append(nearest)
        current_pos = nearest["position"][:2]
        unvisited.remove(nearest)
        
    return [start_point] + ordered + [end_point]

def calculate_path_stats(waypoints):
    """计算路径的详细统计数据"""
    stats = {
        "total_distance": 0.0,
        "segments": [],
        "total_points": len(waypoints)
    }
    
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]["position"][:2]
        p2 = waypoints[i+1]["position"][:2]
        dist = euclidean_distance(p1, p2)
        stats["total_distance"] += dist
        stats["segments"].append({
            "from": waypoints[i]["point_name"],
            "to": waypoints[i+1]["point_name"],
            "distance_m": round(dist, 2)
        })
        
    stats["total_distance"] = round(stats["total_distance"], 2)
    return stats

# ======================
# 3. LLM 交互 (保持不变)
# ======================
def build_prompt(point_dict, last_log):
    task_points = {k: v for k, v in point_dict.items() if k not in ["入口", "出口"]}
    
    map_lines = []
    for name, p in task_points.items():
        task_desc = {0: "无", 1: "检测/取样", 2: "设备检查/维护"}.get(p["arm_task"], "未知")
        # 加入描述信息，让 LLM 更懂业务
        desc = p.get("desc", "")
        map_lines.append(f"- {name}: ({p['x']:.1f}, {p['y']:.1f}) [{task_desc}] ({desc})")
    
    log_info = "无历史记录"
    if last_log.get("visited_points"):
        log_lines = []
        for pt in last_log["visited_points"]:
            status = last_log["tasks_executed"].get(pt, "未知")
            log_lines.append(f"- {pt}: {status}")
        log_info = "\n".join(log_lines)
        if last_log.get("anomalies"):
            log_info += "\n⚠️ 异常:\n" + "\n".join(f"  • {a}" for a in last_log["anomalies"])

    prompt = f"""你是一个智能水处理厂巡检调度专家。
【厂区地图点位】
{chr(10).join(map_lines)}

【上次巡检日志】
{log_info}

【调度规则】
1. 优先重试上次失败 (failed) 的任务点。
2. 成功的点本次可跳过，除非是周期性强制检查点。
3. 路径必须从 "入口" 开始，到 "出口" 结束。
4. 输出严格 JSON 格式，不要 Markdown 标记。
   格式示例：
   {{
     "decision_reason": "上次送水泵检查失败，需优先重试。",
     "inspection_plan": [
       {{"point_name": "巡检点8_送水泵", "action": "task_2"}},
       {{"point_name": "巡检点6_消毒池", "action": "task_1"}}
     ]
   }}
"""
    return prompt

def call_llm(prompt):
    try:
        response = Generation.call(
            model="qwen3-max",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=0.1
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # 简单的 JSON 提取
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        else:
            print(f"API Error: {response.code}")
    except Exception as e:
        print(f"LLM Call Failed: {e}")
    return None

# ======================
# 4. 新增：专业地图绘制功能
# ======================
def draw_water_plant_map(waypoints, output_file):
    """
    绘制水处理厂巡检路线图
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f0f0f0') # 浅灰背景
    
    # 1. 绘制网格背景 (模拟地砖)
    ax.grid(True, which='both', color='#cccccc', linewidth=0.5, linestyle='-')
    ax.set_xticks(range(0, 22, 2))
    ax.set_yticks(range(0, 16, 2))
    ax.set_axisbelow(True)

    # 2. 绘制障碍区域 (带斜线填充)
    for obs in MAP_DATA["obstacles"]:
        rect = patches.Rectangle(
            (obs["x"], obs["y"]), obs["w"], obs["h"],
            linewidth=2, edgecolor='#d9534f', facecolor='#f8d7da',
            hatch='///', label='障碍区域' if obs == MAP_DATA["obstacles"][0] else ""
        )
        ax.add_patch(rect)
        # 标注文字
        ax.text(obs["x"] + obs["w"]/2, obs["y"] + obs["h"]/2, obs["name"],
                ha='center', va='center', color='#d9534f', fontsize=9, fontweight='bold')

    # 3. 绘制主要设备轮廓 (简化示意图)
    # 沉淀池
    ax.add_patch(patches.Rectangle((6, 12), 5, 3, fill=False, edgecolor='#0275d8', linewidth=2))
    ax.text(8.5, 13.5, "沉淀池 1&2", ha='center', va='center', color='#0275d8', fontweight='bold')
    # 过滤床
    ax.add_patch(patches.Rectangle((14, 13), 5, 2.5, fill=False, edgecolor='#0275d8', linewidth=2))
    ax.text(16.5, 14.25, "过滤床", ha='center', va='center', color='#0275d8', fontweight='bold')
    # 消毒池 & 清水池
    ax.add_patch(patches.Rectangle((13, 6.5), 3, 2, fill=False, edgecolor='#0275d8', linewidth=2))
    ax.text(14.5, 7.5, "消毒池", ha='center', va='center', color='#0275d8', fontsize=8)
    ax.add_patch(patches.Rectangle((17, 6.5), 3, 2, fill=False, edgecolor='#0275d8', linewidth=2))
    ax.text(18.5, 7.5, "清水池", ha='center', va='center', color='#0275d8', fontsize=8)
    # 送水泵
    ax.add_patch(patches.Rectangle((14, 2), 6, 1.5, fill=False, edgecolor='#0275d8', linewidth=2))
    ax.text(17, 2.75, "送水泵组", ha='center', va='center', color='#0275d8', fontsize=8)

    # 4. 绘制路径连线
    if len(waypoints) > 1:
        xs = [p["position"][0] for p in waypoints]
        ys = [p["position"][1] for p in waypoints]
        
        # 绘制折线
        ax.plot(xs, ys, color='#28a745', linewidth=2.5, linestyle='--', marker='o', markersize=6, label='巡检路线')
        
        # 绘制箭头指示方向
        for i in range(len(xs) - 1):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            dist = math.hypot(dx, dy)
            if dist > 0.5: # 只有距离够远才画箭头
                arrow_x = xs[i] + dx * 0.6
                arrow_y = ys[i] + dy * 0.6
                ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(arrow_x, arrow_y),
                            arrowprops=dict(arrowstyle='->', color='#28a745', lw=2))

    # 5. 绘制关键点标记
    for i, wp in enumerate(waypoints):
        name = wp["point_name"]
        x, y = wp["position"][0], wp["position"][1]
        
        if name == "入口":
            color = '#28a745' # 绿色
            label = "🚩 入口"
            zorder = 10
        elif name == "出口":
            color = '#dc3545' # 红色
            label = "🏁 出口"
            zorder = 10
        else:
            # 任务点
            task_type = wp.get("arm_task_type", 0)
            color = '#ffc107' if task_type == 2 else '#17a2b8' # 黄色或青色
            label = f"📍 {name.replace('巡检点', '')}" # 简化显示
            zorder = 5
            
        ax.scatter(x, y, c=color, s=150, zorder=zorder, edgecolors='white', linewidth=2)
        ax.text(x, y + 0.4, label, ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color))

    # 6. 添加统计信息文本框
    stats = calculate_path_stats(waypoints)
    info_text = (
        f"📊 巡检统计\n"
        f"总里程：{stats['total_distance']:.2f} m\n"
        f"任务点数：{stats['total_points'] - 2}\n"
        f"路径段数：{len(stats['segments'])}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    # 设置图表属性
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 16)
    ax.set_aspect('equal')
    ax.set_title("🏭 智能水处理厂自动巡检路线图", fontsize=16, pad=20)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ 路线图已保存至：{output_file}")
    plt.close()

# ======================
# 主程序入口
# ======================
def main():
    print("🚀 正在启动水处理厂智能巡检计划生成器...")
    
    # 1. 加载数据
    point_dict = get_all_patrol_points()
    
    # 加载历史日志 (如果存在)
    last_log = {"visited_points": [], "tasks_executed": {}, "anomalies": []}
    if os.path.exists(INSPECTION_LOG_JSON):
        with open(INSPECTION_LOG_JSON, "r", encoding="utf-8") as f:
            last_log = json.load(f)
        print(f"📂 已加载历史日志：{len(last_log.get('visited_points', []))} 个记录")
    else:
        print("📝 未找到历史日志，将执行全量巡检。")

    # 2. 调用 LLM 决策
    print("🧠 正在分析历史数据并生成决策...")
    prompt = build_prompt(point_dict, last_log)
    llm_result = call_llm(prompt)
    
    if not llm_result:
        print("❌ LLM 决策失败，使用默认全量巡检计划。")
        # 默认计划：按编号顺序访问所有点
        default_plan = [{"point_name": k, "action": "task_1" if v["arm_task"]==1 else "task_2"} 
                        for k, v in point_dict.items() if k not in ["入口", "出口"]]
        llm_result = {"decision_reason": "默认全量巡检", "inspection_plan": default_plan}
    
    print(f"💡 决策理由：{llm_result.get('decision_reason', '无')}")

    # 3.  enrich 坐标并排序
    middle_points = []
    action_map = {"arrive": "arrive", "task_1": "execute_arm_task_1", "task_2": "execute_arm_task_2"}
    
    for item in llm_result.get("inspection_plan", []):
        name = item.get("point_name")
        if name and name in point_dict and name not in ["入口", "出口"]:
            p = point_dict[name]
            middle_points.append({
                "point_name": name,
                "position": [p["x"], p["y"], p["z"]],
                "action": action_map.get(item.get("action"), "arrive"),
                "arm_task_type": p["arm_task"]
            })
            
    final_path = solve_tsp_with_fixed_ends(middle_points, point_dict)
    
    # 4. 计算详细统计
    path_stats = calculate_path_stats(final_path)
    
    # 5. 生成最终 JSON
    output_data = {
        "timestamp": "2026-02-13T10:00:00Z", # 示例时间
        "decision_reason": llm_result.get("decision_reason", ""),
        "route_summary": {
            "total_distance_m": path_stats["total_distance"],
            "total_waypoints": path_stats["total_points"],
            "segments": path_stats["segments"]
        },
        "inspection_plan": final_path
    }
    
    with open(OUTPUT_PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"💾 巡检计划已保存：{OUTPUT_PLAN_JSON}")
    
    # 6. 绘制地图
    print("🎨 正在绘制巡检路线图...")
    draw_water_plant_map(final_path, OUTPUT_MAP_IMAGE)
    
    # 7. 更新历史日志 (模拟)
    # 在实际系统中，这里应该是在任务执行完后由机器人回传数据更新
    new_log = {
        "visited_points": [p["point_name"] for p in middle_points],
        "tasks_executed": {p["point_name"]: "success" for p in middle_points},
        "anomalies": []
    }
    with open(INSPECTION_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(new_log, f, indent=2, ensure_ascii=False)
        
    print("✅ 所有任务完成！")

if __name__ == "__main__":
    main()
