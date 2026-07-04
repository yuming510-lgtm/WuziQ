# GPT-Codex Session Template

## CONTROL

### USER_CONTROL

- [x] RUN
- [ ] PAUSE
- [ ] STOP

### SESSION_STATUS

CURRENT_STATUS: GPT_TURN

### ROUND

CURRENT_ROUND: 1  
MAX_ROUNDS: 10  
QUESTION_LIMIT_PER_ROUND: 3  

SESSION_GOAL:  
待填写：本次会话要解决的问题和最终产出。

---

## RULES

- GPT 只在 `CURRENT_STATUS = GPT_TURN` 时处理。
- Codex 只在 `CURRENT_STATUS = CODEX_TURN` 或 `CURRENT_STATUS = CODEX_PLAN_REVIEW` 时处理。
- USER_CONTROL 勾选规则：
  - 勾选 `RUN`：继续；
  - 勾选 `PAUSE`：暂停；
  - 勾选 `STOP`：停止；
  - 如果多个选项同时被勾选，按 `STOP > PAUSE > RUN` 优先级处理；
  - 如果没有任何选项被勾选，默认暂停。
- GPT 每轮最多向 Codex 提出 1–3 个问题。
- Codex 回答必须说明信息来源、参考依据或判断依据。
- 不确定的内容必须明确写“不确定”，不要猜。
- 信息足够后：GPT 生成方案草案 → Codex 评估可落地性 → GPT 生成最终方案。
- 如果 ROUND 区域不够，按相同格式继续新增。
- 达到 `MAX_ROUNDS` 后，GPT 必须生成阶段性方案或最终方案。

---

## USER_TOPIC

待填写：用户发起的问题。可以简短，不需要写背景长文。

---

## ROUNDS

### ROUND 1 - GPT QUESTIONS

等待 GPT 填写。

### ROUND 1 - CODEX ANSWERS

等待 Codex 填写。

### ROUND 1 - GPT ANALYSIS

等待 GPT 填写。

---

## IMPLEMENTATION_PLAN_DRAFT

等待 GPT 生成。

---

## CODEX_FEASIBILITY_REVIEW

等待 Codex 评估。

---

## FINAL_PLAN

等待 GPT 定稿。

---

## SESSION_CLOSE_SUMMARY

等待会话结束后填写。
