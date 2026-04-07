#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
巡检计划质量评估器 (LLM-based Evaluator)
- 功能：统计 100 个输出文件中 SRTP 合格率及机械臂参数合规率
- 核心：调用 LLM 对每个 JSON 文件进行逻辑校验
"""
import os
import json
import re
import dashscope
from dashscope import Generation
from tqdm import tqdm  # 进度条美化

# ======================
# 配置
# ======================
INPUT_DIR = "inspection_plans"
EVALUATION_REPORT = "evaluation_report.json"
API_KEY_ENV = "DASHSCOPE_API_KEY"

# 初始化 API
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv(API_KEY_ENV)

if not dashscope.api_key:
    print(f"❌ 错误：未找到环境变量 {API_KEY_ENV}")
    exit(1)

# 地图点位参考数据 (用于验证坐标和任务类型)
REFERENCE_POINTS = {
    "巡检点1_进水pH": {"x": 2.5, "y": 14.5, "task": 1},
    "巡检点2_沉淀池": {"x": 8.5, "y": 13.5, "task": 1},
    "巡检点3_加药系统": {"x": 4.5, "y": 6.5, "task": 2},
    "巡检点4_澄清池": {"x": 10.5, "y": 10.5, "task": 1},
    "巡检点5_过滤床": {"x": 16.5, "y": 14.5, "task": 1},
    "巡检点6_消毒池": {"x": 14.5, "y": 7.5, "task": 1},
    "巡检点7_清水池": {"x": 18.5, "y": 7.5, "task": 2},
    "巡检点8_送水泵": {"x": 16.5, "y": 3.5, "task": 2},
}

def build_eval_prompt(plan_data, source_log_id):
    """构建给 LLM 的评估提示词"""
    
    # 提取源日志中的异常状态 (从 plan 的 source 或文件名推断，这里假设 plan 中包含决策理由)
    # 为了更精准，我们让 LLM 根据 plan 中的 decision_reason 反推是否合理
    
    prompt = f"""你是一个严格的质检员。请评估以下巡检计划的质量。

【背景信息】
- 任务类型定义：Task 1 = 检测/取样, Task 2 = 设备检查/维护
- 机械臂动作映射规则：
  - Task 1 必须映射为 action: "execute_arm_task_1"
  - Task 2 必须映射为 action: "execute_arm_task_2"
  - 起点/终点 action 必须为 "arrive"
- 优先级规则：如果源日志中某点状态为 "failed"，该点在计划中必须排在其他正常点之前（仅次于入口）。

【待评估数据】
源日志 ID: {source_log_id}
计划数据 (JSON):
{json.dumps(plan_data, ensure_ascii=False)}

【评估任务】
请逐步检查并输出严格的 JSON 结果（不要包含 Markdown 标记）：
1. **format_valid**: (Boolean) JSON 结构是否完整？是否包含 "inspection_plan", "route_summary" 等关键字段？
2. **priority_correct**: (Boolean) 
   - 检查 "decision_reason" 和 "inspection_plan"。
   - 如果决策理由提到了“重试失败点”或类似描述，检查该失败点是否在列表最前方（入口之后第一个）？
   - 如果没有失败点，顺序是否合理？
   - 综合判定优先级逻辑是否无误。
3. **arm_params_valid**: (Boolean)
   - 遍历 "inspection_plan" 中的每个点。
   - 检查该点的 "action" 是否与其标准任务类型 (参考背景信息) 严格匹配？
   - 检查坐标是否在合理范围 (0-20)?
4. **errors**: (List[String]) 列出所有发现的具体错误，如果没有则为空列表。
5. **srtp_pass**: (Boolean) 仅当 format_valid 和 priority_correct 都为 True 时，此项为 True。
6. **arm_pass**: (Boolean) 仅当 arm_params_valid 为 True 时，此项为 True。

输出格式示例：
{{
    "format_valid": true,
    "priority_correct": true,
    "arm_params_valid": true,
    "errors": [],
    "srtp_pass": true,
    "arm_pass": true
}}
"""
    return prompt

def call_llm_evaluator(prompt):
    """调用 LLM 进行评估"""
    try:
        response = Generation.call(
            model="qwen3-max", # 使用逻辑能力强的模型
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=0.0 # 降低温度以获得确定性结果
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # 提取 JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        else:
            return {"error": f"API Error: {response.code}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🔍 开始评估巡检计划质量...")
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 目录 '{INPUT_DIR}' 不存在。请先生成计划。")
        return

    plan_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.json')])
    
    if len(plan_files) == 0:
        print("❌ 未找到任何计划文件。")
        return

    print(f"📂 发现 {len(plan_files)} 个计划文件，开始逐一向 LLM 提问...\n")

    stats = {
        "total": len(plan_files),
        "srtp_success": 0,
        "arm_success": 0,
        "details": []
    }

    # 使用 tqdm 显示进度
    for filename in tqdm(plan_files, desc="评估进度"):
        filepath = os.path.join(INPUT_DIR, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)
            
            # 从文件名提取源日志 ID (例如 plan_LOG_....json -> LOG_...)
            source_id = filename.replace("plan_", "").replace(".json", "")
            
            # 构建提示词并调用 LLM
            prompt = build_eval_prompt(plan_data, source_id)
            result = call_llm_evaluator(prompt)
            
            # 处理结果
            if "error" in result:
                eval_record = {
                    "file": filename,
                    "status": "eval_failed",
                    "reason": result["error"],
                    "srtp": False,
                    "arm": False
                }
            else:
                srtp = result.get("srtp_pass", False)
                arm = result.get("arm_pass", False)
                
                if srtp:
                    stats["srtp_success"] += 1
                if arm:
                    stats["arm_success"] += 1
                    
                eval_record = {
                    "file": filename,
                    "status": "success",
                    "srtp": srtp,
                    "arm": arm,
                    "llm_reasoning": result.get("errors", [])
                }
            
            stats["details"].append(eval_record)
            
        except Exception as e:
            stats["details"].append({
                "file": filename,
                "status": "read_error",
                "reason": str(e),
                "srtp": False,
                "arm": False
            })

    # 计算最终比率
    srtp_rate = (stats["srtp_success"] / stats["total"]) * 100
    arm_rate = (stats["arm_success"] / stats["total"]) * 100

    # 打印报告
    print("\n" + "="*50)
    print("📊 评估报告总结")
    print("="*50)
    print(f"总样本数：{stats['total']}")
    print(f"✅ SRTP 合格数 (格式+优先级): {stats['srtp_success']} ({srtp_rate:.2f}%)")
    print(f"✅ 机械臂参数合规数：{stats['arm_success']} ({arm_rate:.2f}%)")
    print("="*50)
    
    # 保存详细报告
    final_report = {
        "summary": {
            "total_samples": stats["total"],
            "srtp_count": stats["srtp_success"],
            "srtp_percentage": round(srtp_rate, 2),
            "arm_param_count": stats["arm_success"],
            "arm_param_percentage": round(arm_rate, 2)
        },
        "details": stats["details"]
    }
    
    with open(EVALUATION_REPORT, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    
    print(f"💾 详细评估报告已保存至：{EVALUATION_REPORT}")

if __name__ == "__main__":
    main()