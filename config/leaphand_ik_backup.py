# LeapHand IK settings backup - 2026-02-04
# 在锁定 PIP joints 实验前的设置
# 如需回滚，将这些值复制回 vis_denoise.py 和 test_diff_v3.py

LEAPHAND_TIP_MAPPING_BACKUP = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
    # thumb_fingertip 不使用 extra_thumb_tip_head (Z 偏移方向相反)
}

LEAPHAND_LINK_WEIGHTS_VIS_BACKUP = {
    'palm_lower': 3.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 3.0,
}

LEAPHAND_LINK_WEIGHTS_TEST_BACKUP = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 0.8,
}

# 无锁定关节 (原始状态)
LEAPHAND_LOCKED_JOINTS_BACKUP = []
