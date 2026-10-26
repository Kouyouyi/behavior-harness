"""Deterministic presentation fixture, never an executable robot skill plan."""

RESULT_AFTER = 3.2
DECISIONS = [
    '先定位玩偶并检查桌面；目标和可接近位置明确后，才开始靠近。',
    '先调整到抓取工作区；到达并确认可触及时，才允许伸手。',
    '抓取后检查握持与抬升反馈；确认物体被拿起后，再开始搬运。',
    '保持握持并检查移动路径；到达沙发且物体稳定后，再执行放置。',
    '确认放置表面后缓慢放下；收到释放反馈，再撤回手臂。',
    '重新观察物体位置与稳定状态；用独立结果检查结束本次演示。',
]
RESULTS = [
    {'target': 'mickey', 'surface': 'table', 'detected': True},
    {'arrived': True, 'workspace_reachable': True},
    {'grasp_confirmed': True, 'object_lifted': True},
    {'destination': 'sofa', 'arrived': True, 'grasp_stable': True},
    {'released': True, 'arm_retracted': True},
    {'object_on_sofa': True, 'stable': True, 'physical_verification': False},
]


def matches(text):
    return (('米老鼠' in text or 'mickey' in text.lower() or ('米老师' in text and '玩偶' in text))
            and ('沙发' in text or 'sofa' in text.lower()))
