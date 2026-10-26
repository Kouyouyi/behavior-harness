'use strict';

const icons = {
  camera: '<rect x="3" y="6" width="18" height="14" rx="3"/><path d="m8 6 2-3h4l2 3"/><circle cx="12" cy="13" r="4"/>',
  expand: '<path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  terminal: '<path d="m5 7 5 5-5 5M13 17h6"/>',
  mic: '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/>',
  send: '<path d="M12 19V5m-6 6 6-6 6 6"/>',
  sparkles: '<path d="m12 3 2.3 6.7L21 12l-6.7 2.3L12 21l-2.3-6.7L3 12l6.7-2.3L12 3ZM20 2v4M18 4h4"/>',
};
const $ = id => document.getElementById(id);
const ui = {snapshot: null, connected: false, historyKey: '', planKey: '', toolKey: '', outputId: null, output: [], revealed: 0, motors: new Map()};
const groups = [['left_arm', '左臂', 'LEFT ARM / 07'], ['right_arm', '右臂', 'RIGHT ARM / 07'], ['left_leg', '左腿', 'LEFT LEG / 06'], ['right_leg', '右腿', 'RIGHT LEG / 06'], ['torso', '腰部', 'WAIST / 03']];
const jointLabels = {hip_pitch: '髋俯仰', hip_roll: '髋侧摆', hip_yaw: '髋旋转', knee: '膝关节', ankle_pitch: '踝俯仰', ankle_roll: '踝侧摆', waist_yaw: '腰旋转', waist_roll: '腰侧摆', waist_pitch: '腰俯仰', shoulder_pitch: '肩俯仰', shoulder_roll: '肩侧摆', shoulder_yaw: '肩旋转', elbow: '肘关节', wrist_roll: '腕侧摆', wrist_pitch: '腕俯仰', wrist_yaw: '腕旋转'};
const stateLabels = {idle: '待命', running: '执行中', paused: '已暂停', completed: '演示完成', cancelled: '已取消', interrupted: '已中断', failed: '失败', waiting: '等待反馈', unknown: '状态未知'};
function node(tag, className = '', text) { const el = document.createElement(tag); el.className = className; if (text !== undefined) el.textContent = text; return el; }
function icon(name) { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.sparkles}</svg>`; }
function symbol(name, className = '') { const el = node('span', className); el.innerHTML = icon(name); return el; }
function show(id, visible) { $(id).classList.toggle('hidden', !visible); }
function pad(value) { return String(value).padStart(2, '0'); }
function degrees(value) { return Number.isFinite(value) ? value.toFixed(1) : '—'; }
function jointName(name, includeSide = false) { const side = includeSide ? name.startsWith('left_') ? '左' : name.startsWith('right_') ? '右' : '' : ''; return side + (jointLabels[name.replace(/^(left_|right_)/, '').replace(/_joint$/, '')] || name); }
function timeLabel(value) { const date = new Date(typeof value === 'number' ? value * 1000 : value); return Number.isNaN(date.getTime()) ? '—' : date.toLocaleTimeString('zh-CN', {hour12: false}); }
document.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = icon(el.dataset.icon); });

function fitBoard() {
  const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
  const board = $('wallboard');
  board.style.transform = `scale(${scale})`;
  board.style.left = `${(window.innerWidth - 1920 * scale) / 2}px`;
  board.style.top = `${(window.innerHeight - 1080 * scale) / 2}px`;
}
window.addEventListener('resize', fitBoard);
fitBoard();
function updateClock() { const now = new Date(); $('clock').textContent = now.toLocaleTimeString('zh-CN', {hour12: false}); $('date').textContent = `${now.getFullYear()}.${pad(now.getMonth() + 1)}.${pad(now.getDate())}`; }
updateClock(); setInterval(updateClock, 1000);

function duration(value) { return Number.isFinite(value) ? `${pad(Math.floor(value / 60))}:${pad(Math.floor(value % 60))}` : '—'; }
function renderTask(data) {
  const task = data.conversation?.task || data.task;
  const status = task?.status || 'idle';
  $('task-title').textContent = task?.instruction || '等待用户下达任务';
  $('task-title').title = task?.instruction || '';
  $('task-status').textContent = status === 'completed' && data.mode !== 'demo' ? '已完成' : stateLabels[status] || '状态未知';
  $('task-meta').textContent = data.mode === 'demo' ? '模拟演示 · 指令与执行状态同步展示' : 'Agent 任务记录 · 按事件更新';
  $('task-activity').textContent = task?.current_step || data.conversation?.tool?.title || '等待 Agent 事件';
  $('task-elapsed').textContent = duration(task?.elapsed_s);
}

function buildMotors(joints) {
  if (ui.motors.size) return;
  $('motor-groups').replaceChildren(...groups.map(([key, label, english]) => {
    const section = node('section', 'motor-group ' + key);
    const heading = node('div', 'group-heading');
    const title = node('h3', '', label); title.append(node('small', '', english));
    heading.append(title, node('span', '', '外壳 / 绕组 °C'));
    const list = node('div', 'motor-list');
    joints.filter(j => j.group === key).sort((a, b) => a.id - b.id).forEach(j => {
      const row = node('div', 'motor unknown'); row.dataset.motorId = j.id;
      const name = node('span', 'motor-name'); const labelNode = node('span', '', jointName(j.name)); const stateNode = node('small'); name.append(labelNode, stateNode);
      const values = node('span', 'motor-values'); const caseValue = node('small', '', '—'); const windingValue = node('span', '', '—'); values.append(caseValue, node('span', 'slash', '/'), windingValue);
      row.append(node('span', 'motor-id', pad(j.id)), name, values); list.append(row);
      ui.motors.set(j.id, {row, caseValue, windingValue, stateNode});
    });
    section.append(heading, list); return section;
  }));
}

function renderThermal(data) {
  const thermal = data.telemetry;
  buildMotors(thermal.joints);
  const counts = {normal: 0, warning: 0, critical: 0, unknown: 0};
  const alerts = [];
  let hottest = null;
  thermal.joints.forEach(j => {
    const level = ui.connected && j.fresh && ['normal', 'warning', 'critical'].includes(j.level) ? j.level : 'unknown';
    counts[level]++;
    const display = ui.motors.get(j.id);
    display.row.className = 'motor ' + level;
    display.row.dataset.level = level;
    display.caseValue.textContent = degrees(j.case_c);
    display.windingValue.textContent = degrees(j.winding_c);
    const state = level === 'unknown' ? '失效' : j.reason === 'motor_fault' ? '故障' : j.reason === 'alarm_latched' ? '保持' : level === 'critical' ? '过热' : level === 'warning' ? '预警' : '';
    display.stateNode.textContent = state;
    display.row.setAttribute('aria-label', `${jointName(j.name, true)}：外壳 ${degrees(j.case_c)}，绕组 ${degrees(j.winding_c)} 摄氏度，${state || '正常'}`);
    if (level !== 'unknown' && Number.isFinite(j.max_c) && (!hottest || j.max_c > hottest.max_c)) hottest = j;
    if (level === 'warning' || level === 'critical') alerts.push({...j, displayLevel: level});
  });
  for (const key of Object.keys(counts)) $(key + '-count').textContent = pad(counts[key]);
  $('max-temp').textContent = hottest ? degrees(hottest.max_c) : '—';
  $('hottest-joint').textContent = hottest ? jointName(hottest.name, true) : '等待有效温度数据';
  $('thermal-source').textContent = data.mode === 'demo' ? '模拟遥测 · 每秒刷新' : thermal.fresh && ui.connected ? '实时遥测' : '遥测失效';
  $('thermal-freshness').textContent = data.mode === 'demo' ? 'DEMO' : thermal.age_s === null ? 'NO DATA' : `${Math.max(0, thermal.age_s).toFixed(1)} s`;
  const alert = $('thermal-alert');
  alert.className = 'thermal-alert ' + (counts.critical ? 'critical' : counts.warning ? 'warning' : counts.unknown ? 'unknown' : 'normal');
  alerts.sort((a, b) => Number(b.displayLevel === 'critical') - Number(a.displayLevel === 'critical') || b.max_c - a.max_c);
  const missing = counts.unknown ? ` · ${counts.unknown} 个数据失效` : '';
  $('alert-text').textContent = alerts.length ? alerts.slice(0, 3).map(j => `${jointName(j.name, true)} ${degrees(j.max_c)}°${j.reason === 'alarm_latched' ? ' 告警保持' : j.reason === 'motor_fault' ? ' 故障' : ''}`).join(' · ') + missing : counts.unknown ? `${counts.unknown} 个电机数据缺失或过期` : '全部 29 个电机温度正常';
  $('red-threshold').textContent = thermal.thermal_rule === 'strict91' ? '告警 >91°C / 电机故障' : '告警 ≥90°C / 电机故障';
  $('thermal-rule-note').textContent = thermal.thermal_rule === 'strict91' ? 'STRICT91 · 85°C 起预警，大于 91°C 告警 · 灰色为失效数据' : 'NATIVE90 · 踝部告警低于 85°C 解除 · 灰色为失效数据';
}

function positionAnnotation() {
  const image = $('camera-image'); if (!image.naturalWidth || !image.clientWidth) return;
  const scale = Math.max(image.clientWidth / image.naturalWidth, image.clientHeight / image.naturalHeight);
  const width = image.naturalWidth * scale, height = image.naturalHeight * scale;
  Object.assign($('target-box').style, {left: `${(image.clientWidth - width) / 2 + width * .28}px`, top: `${(image.clientHeight - height) / 2 + height * .34}px`, width: `${width * .3}px`, height: `${height * .4}px`});
}
new ResizeObserver(positionAnnotation).observe($('camera-stage'));
$('camera-image').addEventListener('load', () => { positionAnnotation(); show('camera-empty', false); });
$('camera-image').addEventListener('error', () => { show('camera-empty', true); $('camera-label').textContent = '图像不可用'; });

function renderCamera(data) {
  const camera = data.camera, demo = data.mode === 'demo';
  if (camera.url && $('camera-image').getAttribute('src') !== camera.url) $('camera-image').src = camera.url;
  show('camera-image', !!camera.url); show('camera-empty', !camera.url);
  show('target-box', demo && !!camera.url);
  const stale = !!camera.url && (!ui.connected || (!demo && !camera.fresh));
  show('camera-stale', stale); $('camera-stage').classList.toggle('stale', stale);
  $('camera-source').textContent = demo ? 'RECORDED SAMPLE / E050' : camera.fresh ? 'LIVE OBSERVATION' : 'WAITING FOR SIGNAL';
  $('camera-label').textContent = demo ? '历史实拍样本 · 非直播' : camera.fresh ? '实时画面' : '等待有效观测';
  $('camera-caption').textContent = demo ? '目标：桌面上的米老鼠' : camera.fresh ? '机器人正在观察环境' : '等待新的观测';
  $('camera-note').textContent = demo ? '历史照片与演示标注 · 不代表当前任务的实时结果' : '独立相机数据源 · 按源时间校验有效性';
  $('frame-detail').textContent = demo ? '640 × 360' : camera.age_s === null ? 'NO FEED' : `AGE ${Math.max(0, camera.age_s).toFixed(1)}s`;
  $('vision-footer-text').textContent = demo ? '观测样本 E050 · 目标框为演示标注' : camera.fresh ? 'RGB-D 数据链路已连接' : '相机数据缺失或过期';
  $('robot-subtitle').textContent = data.mode === 'demo' ? 'UNITREE G1 · DEMONSTRATION' : data.robot.status === 'observing' ? 'UNITREE G1 · READ-ONLY' : 'UNITREE G1 · WAITING FOR TELEMETRY';
  positionAnnotation();
}

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const phaseLabels = {idle: '等待任务', thinking: '正在思考', tool: '正在调用工具', responding: '正在输出', paused: '已暂停', completed: '演示完成', cancelled: '已取消', interrupted: '已中断', observing: 'Agent 公开输出', replied: '已回复'};
function followConversation() {
  const thread = $('conversation-thread');
  thread.scrollTop = thread.scrollHeight;
}
function updateOutput() {
  const conversation = ui.snapshot?.conversation;
  if (!conversation) return;
  if (reducedMotion.matches) ui.revealed = ui.output.length;
  else if (ui.connected && conversation.phase !== 'paused') ui.revealed = Math.min(ui.output.length, ui.revealed + 2);
  const text = ui.output.slice(0, ui.revealed).join('');
  if ($('agent-text').textContent !== text) { $('agent-text').textContent = text; followConversation(); }
  show('typing-caret', ui.connected && conversation.phase !== 'paused' && (ui.revealed < ui.output.length || !!conversation.output?.streaming));
}
setInterval(updateOutput, 40);

function renderConversation(data) {
  const conversation = data.conversation;
  if (!conversation) return;
  const history = conversation.history || [];
  const historyKey = JSON.stringify(history);
  if (historyKey !== ui.historyKey) {
    ui.historyKey = historyKey;
    $('conversation-history').replaceChildren(...history.map(message => {
      const entry = node('article', 'history-message ' + (message.role === 'user' ? 'user' : 'assistant'));
      entry.append(node('span', 'history-meta', message.role === 'user' ? '你' : 'G1 Agent'), node('p', '', message.text));
      return entry;
    }));
  }
  const plan = Array.isArray(conversation.plan) ? conversation.plan : [];
  show('agent-plan', plan.length > 0);
  const planKey = JSON.stringify(plan);
  if (planKey !== ui.planKey) {
    ui.planKey = planKey;
    $('plan-steps').replaceChildren(...plan.map((step, index) => {
      const status = ['pending', 'running', 'completed', 'failed'].includes(step.status) ? step.status : 'pending';
      const entry = node('div', 'plan-step ' + status);
      const badge = node('i', '', String(index + 1));
      if (status === 'completed') badge.innerHTML = icon('check');
      entry.append(badge, node('span', '', step.label));
      if (status === 'running') entry.setAttribute('aria-current', 'step');
      return entry;
    }));
    $('plan-count').textContent = `${plan.filter(s => s.status === 'completed').length} / ${plan.length}`;
  }
  $('plan-note').textContent = conversation.plan_note || '';
  $('plan-source').textContent = data.mode === 'demo' ? '样例模拟拆解' : 'Agent 计划记录';
  $('user-text').textContent = conversation.user?.text || '等待用户指令';
  $('user-time').textContent = conversation.user?.time ? timeLabel(conversation.user.time) : '—';
  $('chat-mode').textContent = data.mode === 'demo' ? '演示会话' : data.capabilities?.chat ? 'Agent 消息与执行记录' : 'Agent 事件记录';
  const active = ui.connected && ['thinking', 'tool', 'responding'].includes(conversation.phase);
  $('agent-work').classList.toggle('active', active);
  $('agent-status').textContent = ui.connected ? phaseLabels[conversation.phase] || '等待反馈' : '连接中断 · 已停止更新';
  $('work-elapsed').textContent = Number.isFinite(conversation.elapsed_s) ? `本次处理 ${conversation.elapsed_s.toFixed(1)} s` : '';
  $('decision-text').textContent = conversation.summary || '依据观测与工具反馈更新公开摘要。';
  $('conversation-caption').textContent = !ui.connected ? '等待重新连接' : conversation.phase === 'paused' ? '任务暂停 · 输出已冻结' : data.mode === 'demo' ? '模拟过程 · 自动跟随最新输出' : '自动跟随最新记录';
  const tool = conversation.tool;
  show('tool-card', !!tool);
  const key = JSON.stringify(tool);
  if (key !== ui.toolKey) {
    ui.toolKey = key;
    $('tool-name').textContent = typeof tool?.tool === 'string' ? tool.tool : 'MCP';
    $('tool-status').textContent = data.mode === 'demo' ? tool?.status === 'completed' ? '模拟返回 ✓' : '模拟调用中' : tool?.status === 'completed' ? '已返回' : tool?.status === 'failed' ? '调用失败' : '调用记录';
    $('tool-arguments').textContent = tool?.details?.arguments ? JSON.stringify(tool.details.arguments) : tool?.text || '';
    show('tool-result', !!tool?.result);
    $('tool-result').textContent = tool?.result ? '返回  ' + JSON.stringify(tool.result) : '';
    followConversation();
  }
  const output = conversation.output || {};
  const target = Array.from(typeof output.text === 'string' ? output.text : '');
  if (output.id !== ui.outputId || !target.join('').startsWith(ui.output.slice(0, ui.revealed).join(''))) {
    ui.outputId = output.id; ui.revealed = 0;
  }
  ui.output = target;
  updateOutput();
}

function render(data) {
  ui.snapshot = data;
  $('mode-badge').textContent = data.mode === 'demo' ? (data.presentation?.autoplay ? '自动演示 · 模拟数据' : '演示模式 · 模拟数据') : data.capabilities?.chat ? '实机观测 · Agent 消息' : '实机观测 · 只读';
  $('footer-source').textContent = data.mode === 'demo' ? '演示任务 / 合成温度 / 历史照片 · 当前不连接真实模型与运动控制' : data.capabilities?.chat ? '实机观测 · 输入发送至 Agent 收件箱，接收不代表动作完成' : '实机只读观测 · 全部状态以源时间戳校验，不发送机器人动作';
  renderTask(data); renderThermal(data); renderCamera(data); renderConversation(data);
  document.dispatchEvent(new Event('console-state'));
}

async function refresh() {
  try {
    const response = await fetch('/api/v1/state', {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error(String(response.status));
    const data = await response.json(); ui.connected = true;
    $('connection-status').className = 'link-state'; $('connection-status').replaceChildren(node('i'), document.createTextNode('展示链路在线'));
    show('connection-error', false); render(data);
  } catch (error) {
    ui.connected = false;
    $('connection-status').className = 'link-state offline'; $('connection-status').replaceChildren(node('i'), document.createTextNode('连接中断'));
    $('connection-error').textContent = error.message === '401' ? '凭据已失效 · 请使用启动地址重新打开大屏' : '数据连接中断 · 当前值已标记过期';
    show('connection-error', true);
    if (ui.snapshot) { renderThermal(ui.snapshot); renderCamera(ui.snapshot); renderConversation(ui.snapshot); }
    document.dispatchEvent(new Event('console-state'));
  }
}
async function poll() { await refresh(); setTimeout(poll, 250); }
$('fullscreen-button').addEventListener('click', async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { $('fullscreen-button').title = '请使用浏览器 F11 全屏'; } });
poll();
