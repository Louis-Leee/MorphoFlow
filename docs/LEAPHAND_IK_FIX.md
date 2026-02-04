# LeapHand IK Fix: 诊断与修复文档 (v3)

## 问题概述

LeapHand 在 Pyroki IK 优化后，绿色 mesh 无法与 diffusion model 输出的彩色点云对齐：
1. **thumb_fingertip** - 大拇指指尖无法 align
2. **palm_lower** - 手掌也无法很好地 align

**参考对比**: Allegro 的 IK 优化结果完美

---

## 根本原因分析

### Allegro vs LeapHand 关键差异

| 方面 | Allegro (完美) | LeapHand (问题) |
|------|---------------|-----------------|
| **指尖关节类型** | Fixed joint | Revolute joint |
| **雅可比矩阵** | 干净的位置约束 | 有旋转关节干扰 |
| **链接权重** | 统一权重 | 需要 per-link 权重 |
| **特殊处理** | 无 | thumb 偏移方向相反 |

### URDF 结构对比

**Allegro URDF**:
```xml
<!-- 指尖用 FIXED joint -->
<joint name="joint_3.0_tip" type="fixed">
    <parent link="link_3.0"/>
    <child link="link_3.0_tip"/>
</joint>
```

**LeapHand URDF**:
```xml
<!-- 指尖用 REVOLUTE joint (问题根源) -->
<joint name="joint 3" type="revolute">
    <parent link="dip"/>
    <child link="fingertip"/>
    <axis xyz="0 0 -1"/>
</joint>

<!-- extra tip 作为 fixed joint (workaround) -->
<joint name="index_tip" type="fixed">
    <origin xyz="0 -0.048 0.015"/>  <!-- Z = +0.015 -->
</joint>

<!-- 大拇指的 extra tip Z 方向相反！ -->
<joint name="thumb_tip" type="fixed">
    <origin xyz="0 -0.06 -0.015"/>  <!-- Z = -0.015 -->
</joint>
```

---

## 当前实现状态

### 已完成的修改

1. **vis_denoise.py** - IK 实验模式框架:
   - `IK_MODE_DEFAULTS`: baseline, thumb_extra, palm_anchor, combo
   - `_resolve_ik_cfg()`: 解析配置
   - `_derive_base_pos_from_palm()`: 从 palm 推导 base 位置
   - `_fk_link_positions()`: FK 计算
   - `_write_ik_report()`: 误差报告生成

2. **config/vis_denoise.yaml** - IK 配置:
   ```yaml
   vis:
     ik:
       mode: "baseline"  # baseline | thumb_extra | palm_anchor | combo
       force_recompute: true
       report_path: "reports/ik_report_baseline.md"
   ```

3. **pyroki_ik.py** - per-link 权重支持

4. **LeapHand 特殊处理**:
   ```python
   LEAPHAND_TIP_MAPPING = {
       'fingertip': 'extra_index_tip_head',
       'fingertip_2': 'extra_middle_tip_head',
       'fingertip_3': 'extra_ring_tip_head',
       # thumb_fingertip 保持原样 (Z 偏移方向相反)
   }

   LEAPHAND_LINK_WEIGHTS = {
       'palm_lower': 2.0,
       'thumb_fingertip': 0.8,
   }
   ```

---

## 实验模式说明

| 模式 | thumb_extra | palm_anchor | 描述 |
|------|-------------|-------------|------|
| baseline | False | False | 当前默认行为 |
| thumb_extra | True | False | thumb 也用 extra_thumb_tip_head |
| palm_anchor | False | True | 添加 base 锚点约束 palm 姿态 |
| combo | True | True | 两者结合 |

---

## 执行实验步骤

### Step 1: 运行 baseline 模式
```bash
# 配置已设置: mode: "baseline", report_path: "reports/ik_report_baseline.md"
python vis_denoise.py --config config/vis_denoise.yaml
```

### Step 2-4: 运行其他模式
```bash
# 修改 config/vis_denoise.yaml 中的 mode 和 report_path
# mode: thumb_extra / palm_anchor / combo
python vis_denoise.py --config config/vis_denoise.yaml --load outputs/vis_cache.pt
```

### Step 5: 对比报告
查看 `reports/` 目录下的四个报告:
- `overall_mean_mm`: 越小越好
- `worst_links_mm`: 关注 thumb_fingertip 和 palm_lower

---

## 关键文件

| 文件 | 功能 |
|------|------|
| `vis_denoise.py` | IK 实验框架主逻辑 |
| `config/vis_denoise.yaml` | IK 模式配置 |
| `config/vis_denoise.baseline.yaml` | 原始配置备份 |
| `pyroki_ik.py` | IK 求解器 (per-link 权重) |
| `utils/palm_centric.py` | `get_palm_link_name()` |
| `utils/optimization.py` | `process_transform()` |

---

## 修改范围限制

所有 LeapHand 相关修改都有条件保护:
```python
if robot_name == 'leaphand':
    # LeapHand 特殊处理
```

不影响其他机器人 (Allegro, Barrett, ShadowHand 等)

---

## 历史版本

### v1: 使用 extra_*_tip_head 替代 fingertip_*
- 问题: thumb 的 Z 偏移方向相反，导致 thumb 更收拢

### v2: 移除 thumb 映射 + per-link 权重
- 改进: thumb 不再使用 extra_thumb_tip_head
- 添加: palm_lower=2.0, thumb_fingertip=0.8

### v3: 实验模式框架 (当前)
- 添加: 四种 IK 模式 (baseline, thumb_extra, palm_anchor, combo)
- 添加: 误差报告生成
- 添加: base 锚点推导
