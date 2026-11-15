const $ = (id) => document.getElementById(id);
const state = { models: [], selectedFile: null };

async function jsonFetch(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail?.error || payload.detail?.message || payload.error || `HTTP ${response.status}`);
  return payload;
}

function setError(message = '') { $('error').textContent = message; }
function selectedModel() { return state.models.find((item) => item.id === $('model').value); }
function updatePrecisionOptions() {
  const model = selectedModel();
  const available = model ? Object.keys(model.precisions) : [];
  [...$('precision').options].forEach((option) => { option.disabled = !available.includes(option.value); });
  if (!available.includes($('precision').value)) $('precision').value = available[0] || 'fp16';
}

async function loadModels() {
  const payload = await jsonFetch('/v1/models');
  state.models = payload.models;
  $('model').replaceChildren(...state.models.map((model) => new Option(`${model.id} · ${model.dataset}`, model.id)));
  updatePrecisionOptions();
  $('run').disabled = !state.models.length;
  if (!state.models.length) setError('No validated models are registered. Mount engines and configure models.json.');
}

async function loadReadiness() {
  try {
    const payload = await jsonFetch('/readyz');
    $('readiness').textContent = `Ready · ${payload.models_loaded} cached`;
    $('readiness').className = 'status status-ok';
  } catch (error) {
    $('readiness').textContent = 'Not ready';
    $('readiness').className = 'status status-bad';
  }
}

function showInput(file) {
  const url = URL.createObjectURL(file);
  $('input-preview').src = url;
  $('input-preview').classList.add('visible');
  $('input-empty').hidden = true;
}

function renderTiming(timing) {
  const entries = Object.entries(timing).filter(([key]) => key !== 'total');
  const max = Math.max(...entries.map(([, value]) => value), 1);
  $('timing-bars').innerHTML = entries.map(([key, value]) => `<div class="bar-row"><span>${key.replaceAll('_', ' ')}</span><div class="bar"><i style="width:${Math.max(2, value / max * 100)}%"></i></div><b>${Number(value).toFixed(2)} ms</b></div>`).join('');
}

function formatBytes(value) {
  if (value == null) return 'N/A';
  const units = ['B', 'MB', 'GB'];
  let number = value;
  let index = 0;
  while (number >= 1024 * 1024 && index < units.length - 1) { number /= 1024 * 1024; index += 1; }
  return `${number.toFixed(1)} ${units[index]}`;
}

function clearResult() {
  $('output-preview').removeAttribute('src');
  $('output-preview').classList.remove('visible');
  $('output-empty').hidden = false;
  $('total-latency').textContent = '—';
  $('inference-latency').textContent = '—';
  $('fps').textContent = 'FPS —';
  $('backend').textContent = 'Backend —';
  $('detection-count').textContent = '—';
  $('confidence').textContent = 'Confidence —';
  $('gpu-load').textContent = 'GPU —';
  $('cpu-load').textContent = 'CPU —';
  $('ram-load').textContent = 'RAM —';
  $('gpu-memory').textContent = 'GPU memory —';
  $('timing-bars').replaceChildren();
  $('provenance').innerHTML = '<dt>Model</dt><dd>—</dd>';
  $('detections').innerHTML = '<tr><td colspan=\"3\" class=\"empty\">No result yet</td></tr>';
}

function renderResult(result) {
  const timing = result.timing_ms;
  $('output-preview').src = `${result.output_image_url}?request_id=${result.request_id}`;
  $('output-preview').classList.add('visible');
  $('output-empty').hidden = true;
  $('total-latency').textContent = `${timing.total.toFixed(2)} ms`;
  $('inference-latency').textContent = `${timing.inference.toFixed(2)} ms`;
  $('fps').textContent = `FPS ${result.throughput.fps_sequential?.toFixed(2) ?? '—'}`;
  $('backend').textContent = `${result.model.backend} · ${result.model.precision.toUpperCase()}`;
  $('detection-count').textContent = result.detections.length;
  const displayThreshold = result.decision?.display_conf ?? 0.3;
  $('confidence').textContent = `Confidence ≥ ${displayThreshold.toFixed(2)} ${confidences.length ? `· ${Math.min(...confidences).toFixed(2)}–${Math.max(...confidences).toFixed(2)}` : '· —'}`;
  $('cpu-load').textContent = `CPU ${result.resources.cpu_percent == null ? 'N/A' : `${result.resources.cpu_percent.toFixed(1)}%`}`;
  $('gpu-load').textContent = `GPU ${result.resources.gpu_percent == null ? 'N/A' : `${result.resources.gpu_percent.toFixed(1)}%`}`;
  $('ram-load').textContent = `RAM ${formatBytes(result.resources.ram_used_bytes)} / ${formatBytes(result.resources.ram_total_bytes)}`;
  $('gpu-memory').textContent = `GPU memory ${formatBytes(result.resources.gpu_memory_used_bytes)} / ${formatBytes(result.resources.gpu_memory_total_bytes)}`;
  renderTiming(timing);
  $('provenance').innerHTML = [
    ['Model', result.model.id], ['Dataset', result.model.dataset], ['Variant', result.model.variant], ['Seed', result.model.seed ?? '—'],
    ['Precision', result.model.precision.toUpperCase()], ['Input', `${result.image.width} × ${result.image.height}`], ['SHA-256', result.model.sha256], ['NMS IoU', result.protocol.nms_iou],
    ['Timing source', result.timing_source],
  ].map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join('');
  $('detections').innerHTML = result.detections.length ? result.detections.map((item) => `<tr><td>${item.class_name}</td><td>${(item.confidence * 100).toFixed(2)}%</td><td>[${item.xyxy.map((value) => value.toFixed(1)).join(', ')}]</td></tr>`).join('') : '<tr><td colspan="3" class="empty">No detections</td></tr>';
}

async function runInference() {
  if (!state.selectedFile) return;
  setError('');
  $('run').disabled = true;
  $('run').textContent = 'Running…';
  const body = new FormData(); body.append('file', state.selectedFile);
  try {
    const result = await jsonFetch(`/v1/models/${encodeURIComponent($('model').value)}/predict?precision=${$('precision').value}`, { method: 'POST', body });
    renderResult(result);
    await loadMetrics();
  } catch (error) { clearResult(); setError(error.message); } finally { $('run').disabled = false; $('run').textContent = 'Run inference'; }
}

async function loadMetrics() {
  try {
    const metrics = await jsonFetch('/v1/metrics');
    $('history').innerHTML = metrics.history.length ? metrics.history.map((item) => `<tr><td>${item.timestamp}</td><td>${item.model_id}</td><td>${item.precision.toUpperCase()}</td><td>${item.detection_count}</td><td>${item.total_ms.toFixed(2)} ms</td><td>${item.ok ? 'OK' : item.error_code}</td></tr>`).join('') : '<tr><td colspan="6" class="empty">No requests yet</td></tr>';
  } catch (_) { /* readiness surface already shows the primary failure */ }
}

$('file').addEventListener('change', (event) => { state.selectedFile = event.target.files[0] || null; if (state.selectedFile) showInput(state.selectedFile); $('run').disabled = !state.selectedFile || !state.models.length; });
$('model').addEventListener('change', updatePrecisionOptions);
$('run').addEventListener('click', runInference);
Promise.all([loadModels(), loadReadiness(), loadMetrics()]).catch((error) => setError(error.message));
setInterval(loadReadiness, 10000);
