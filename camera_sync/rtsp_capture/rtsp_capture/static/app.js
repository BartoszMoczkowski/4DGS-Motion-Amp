/* ==========================================================================
   4DGS Multi-Camera Capture Studio — Frontend Controller
   ========================================================================== */

const state = {
  telemetry: null,
  previewInterval: null,
  previewActive: true,
  cameras: {},
  recording: false,
};

// Preset configurations
const PRESETS = {
  daylight: { wb_mode: 'Manual', red: 50, blue: 50, exp_mode: 'Manual', shutter: '1/100', gain: 50, brightness: 50, contrast: 50 },
  cool: { wb_mode: 'Manual', red: 42, blue: 60, exp_mode: 'Manual', shutter: '1/100', gain: 50, brightness: 50, contrast: 50 },
  warm: { wb_mode: 'Manual', red: 60, blue: 42, exp_mode: 'Manual', shutter: '1/100', gain: 50, brightness: 50, contrast: 50 },
  high_speed: { wb_mode: 'Manual', red: 50, blue: 50, exp_mode: 'Manual', shutter: '1/250', gain: 65, brightness: 55, contrast: 52 },
};

// ==========================================================================
// Initialization
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Set default run name timestamp
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const ts = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  document.getElementById('session-name').value = `run_${ts}`;

  // Start telemetry event stream (SSE)
  connectEventStream();

  // Start preview snapshot polling
  startPreviewPolling();
});

// ==========================================================================
// Telemetry & Event Stream (SSE)
// ==========================================================================

function connectEventStream() {
  const evtSource = new EventSource('/api/events');

  evtSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      state.telemetry = data;
      renderTelemetry(data);
    } catch (err) {
      console.error('Failed to parse SSE event data', err);
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    // Fallback to polling every 1 sec
    setTimeout(connectEventStream, 2000);
  };
}

function renderTelemetry(telem) {
  // Update state badges
  const stateBadge = document.getElementById('rig-state-badge');
  stateBadge.textContent = telem.state;
  stateBadge.className = 'badge ' + (
    telem.state === 'RECORDING' ? 'badge-recording' :
    telem.state === 'PROBING' ? 'badge-probing' : 'badge-idle'
  );

  const mockBadge = document.getElementById('mock-mode-badge');
  if (telem.mock_mode) {
    mockBadge.classList.remove('hidden');
  } else {
    mockBadge.classList.add('hidden');
  }

  const camCountBadge = document.getElementById('cam-count-badge');
  camCountBadge.textContent = `${telem.camera_count} CAMERAS`;

  // Update Recording Button & Timer
  const recordBtn = document.getElementById('record-btn');
  const recordText = document.getElementById('record-btn-text');
  const timerBox = document.getElementById('live-timer-box');
  const timerDigits = document.getElementById('live-elapsed');

  if (telem.state === 'RECORDING') {
    state.recording = true;
    recordBtn.classList.add('recording');
    recordText.textContent = 'Stop Recording';
    timerBox.classList.remove('hidden');

    const sec = telem.elapsed_sec || 0;
    const mins = Math.floor(sec / 60);
    const remainder = (sec % 60).toFixed(1);
    timerDigits.textContent = `${String(mins).padStart(2, '0')}:${String(remainder).padStart(4, '0')}`;
  } else {
    state.recording = false;
    recordBtn.classList.remove('recording');
    recordText.textContent = 'Record';
    timerBox.classList.add('hidden');
  }

  // Render camera cards if not already created
  if (telem.cameras) {
    updateCameraCards(telem.cameras);
  }
}

// ==========================================================================
// Camera Grid & Cards
// ==========================================================================

function updateCameraCards(cameras) {
  const container = document.getElementById('camera-cards-container');

  for (const [cid, cam] of Object.entries(cameras)) {
    let card = document.getElementById(`cam-card-${cid}`);
    if (!card) {
      card = createCameraCard(cid, cam);
      container.appendChild(card);
    }

    // Update dynamic stats
    const dot = card.querySelector('.status-dot');
    if (dot) {
      dot.className = `status-dot ${cam.online ? '' : 'offline'}`;
    }

    const frameVal = card.querySelector('.stat-frames');
    if (frameVal && cam.stream) {
      frameVal.textContent = cam.stream.frames || 0;
    }

    const srVal = card.querySelector('.stat-srs');
    if (srVal && cam.stream) {
      srVal.textContent = cam.stream.srs || 0;
    }

    // Update flashing overlay if hardware is pulsing
    const flashOverlay = card.querySelector('.flash-overlay');
    if (flashOverlay) {
      if (cam.light && cam.light.is_flashing) {
        card.classList.add('flashing');
        flashOverlay.classList.add('active');
      } else {
        card.classList.remove('flashing');
        flashOverlay.classList.remove('active');
      }
    }
  }
}

function createCameraCard(cid, cam) {
  const card = document.createElement('div');
  card.className = 'camera-card';
  card.id = `cam-card-${cid}`;

  const initialRed = cam.color ? cam.color.gain_red : 50;
  const initialBlue = cam.color ? cam.color.gain_blue : 50;
  const initialShutter = cam.exposure ? cam.exposure.shutter : '1/100';

  card.innerHTML = `
    <div class="card-header">
      <div class="card-title-group">
        <span class="status-dot ${cam.online ? '' : 'offline'}"></span>
        <span class="card-cam-id">${cid.toUpperCase()}</span>
        <span class="card-ip">${cam.ip}</span>
      </div>
      <div class="card-header-actions">
        <button class="btn-card-action btn-flash" onclick="flashCamera('${cid}')" title="Flash LED to identify camera in the rig">
          ⚡ Identify
        </button>
        <button class="btn-card-action" onclick="toggleCameraLight('${cid}')" title="Toggle white light illuminator">
          💡 Light
        </button>
      </div>
    </div>

    <div class="preview-container">
      <img id="preview-${cid}" class="preview-img" src="/api/cameras/${cid}/snapshot?t=${Date.now()}" alt="${cid}">
      <div class="preview-overlay-info">${cid.toUpperCase()}</div>
      <div class="flash-overlay">⚡ IDENTIFYING</div>
    </div>

    <div class="card-stats-footer">
      <div class="card-stats-item">
        <span>Frames:</span>
        <span class="val stat-frames">0</span>
      </div>
      <div class="card-stats-item">
        <span>RTCP SRs:</span>
        <span class="val stat-srs">0</span>
      </div>
      <div class="card-stats-item">
        <span class="badge" style="font-size: 9px; padding: 2px 4px;">SYNC READY</span>
      </div>
    </div>

    <div class="card-accordion">
      <div class="accordion-title" onclick="toggleCardTuning('${cid}')">
        <span>⚙️ Fine-Tune Override</span>
        <span id="acc-arrow-${cid}">▼</span>
      </div>
      <div id="acc-body-${cid}" class="accordion-body hidden" style="display:flex; flex-direction:column; gap:8px; margin-top:6px;">
        <div class="slider-row">
          <span class="slider-tag red-tag">Red Gain</span>
          <input type="range" id="tune-red-${cid}" min="0" max="100" value="${initialRed}" oninput="document.getElementById('val-tune-red-${cid}').textContent = this.value">
          <span class="slider-val" id="val-tune-red-${cid}">${initialRed}</span>
        </div>
        <div class="slider-row">
          <span class="slider-tag blue-tag">Blue Gain</span>
          <input type="range" id="tune-blue-${cid}" min="0" max="100" value="${initialBlue}" oninput="document.getElementById('val-tune-blue-${cid}').textContent = this.value">
          <span class="slider-val" id="val-tune-blue-${cid}">${initialBlue}</span>
        </div>
        <div class="control-row">
          <label>Shutter:</label>
          <select id="tune-shutter-${cid}">
            <option value="1/50" ${initialShutter==='1/50'?'selected':''}>1/50 s</option>
            <option value="1/100" ${initialShutter==='1/100'?'selected':''}>1/100 s</option>
            <option value="1/250" ${initialShutter==='1/250'?'selected':''}>1/250 s</option>
            <option value="1/500" ${initialShutter==='1/500'?'selected':''}>1/500 s</option>
          </select>
          <button class="btn btn-sm btn-primary" onclick="saveSingleCameraTuning('${cid}')">Save</button>
        </div>
      </div>
    </div>
  `;

  return card;
}

function toggleCardTuning(cid) {
  const body = document.getElementById(`acc-body-${cid}`);
  const arrow = document.getElementById(`acc-arrow-${cid}`);
  if (body.classList.contains('hidden')) {
    body.classList.remove('hidden');
    arrow.textContent = '▲';
  } else {
    body.classList.add('hidden');
    arrow.textContent = '▼';
  }
}

// ==========================================================================
// Live Preview Snapshots Polling
// ==========================================================================

function startPreviewPolling() {
  if (state.previewInterval) clearInterval(state.previewInterval);

  state.previewInterval = setInterval(() => {
    if (!state.previewActive) return;

    const cards = document.querySelectorAll('.camera-card');
    const now = Date.now();
    cards.forEach((card) => {
      const cid = card.id.replace('cam-card-', '');
      const img = document.getElementById(`preview-${cid}`);
      if (img) {
        img.src = `/api/cameras/${cid}/snapshot?t=${now}`;
      }
    });

    const tsEl = document.getElementById('preview-timestamp');
    if (tsEl) {
      const d = new Date();
      tsEl.textContent = `Updated ${d.toLocaleTimeString()}`;
    }
  }, 600); // ~1.6 FPS
}

function togglePreviewPolling(active) {
  state.previewActive = active;
  if (!active) {
    document.getElementById('preview-timestamp').textContent = 'Previews paused';
  }
}

// ==========================================================================
// Hardware Flash & Identification
// ==========================================================================

async function flashCamera(cid) {
  const card = document.getElementById(`cam-card-${cid}`);
  if (card) {
    card.classList.add('flashing');
    const overlay = card.querySelector('.flash-overlay');
    if (overlay) overlay.classList.add('active');
  }

  try {
    const res = await fetch(`/api/cameras/${cid}/flash`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ duration_sec: 3.0, pulses: 4 }),
    });
    if (res.ok) {
      showToast(`⚡ Flashing light on ${cid.toUpperCase()} for identification`, 'success');
    } else {
      showToast(`Failed to flash ${cid}`, 'error');
    }
  } catch (err) {
    showToast(`Error flashing ${cid}: ${err}`, 'error');
  } finally {
    setTimeout(() => {
      if (card) {
        card.classList.remove('flashing');
        const overlay = card.querySelector('.flash-overlay');
        if (overlay) overlay.classList.remove('active');
      }
    }, 3200);
  }
}

async function flashAllSequence() {
  showToast('⚡ Sequentially flashing all cameras in numerical order...', 'warn');
  const cards = Array.from(document.querySelectorAll('.camera-card'));
  for (let i = 0; i < cards.length; i++) {
    const cid = cards[i].id.replace('cam-card-', '');
    flashCamera(cid);
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

async function toggleCameraLight(cid) {
  try {
    const res = await fetch(`/api/cameras/${cid}/light`, { method: 'POST' });
    const data = await res.json();
    showToast(`💡 Light toggled for ${cid.toUpperCase()}`, 'success');
  } catch (err) {
    showToast(`Light toggle error: ${err}`, 'error');
  }
}

async function batchToggleLights(mode, brightness) {
  const cards = document.querySelectorAll('.camera-card');
  for (const card of cards) {
    const cid = card.id.replace('cam-card-', '');
    fetch(`/api/cameras/${cid}/light?mode=${mode}&brightness=${brightness}`, { method: 'POST' });
  }
  showToast(`Lights set to ${mode} across all cameras`, 'success');
}

// ==========================================================================
// Universal Master Calibration ("Apply to All")
// ==========================================================================

function applyPreset(presetKey) {
  const p = PRESETS[presetKey];
  if (!p) return;

  document.getElementById('master-wb-mode').value = p.wb_mode;
  document.getElementById('master-wb-red').value = p.red;
  document.getElementById('val-wb-red').textContent = p.red;
  document.getElementById('master-wb-blue').value = p.blue;
  document.getElementById('val-wb-blue').textContent = p.blue;

  document.getElementById('master-exp-mode').value = p.exp_mode;
  document.getElementById('master-shutter').value = p.shutter;
  document.getElementById('master-gain-max').value = p.gain;
  document.getElementById('val-exp-gain').textContent = p.gain;

  document.getElementById('master-brightness').value = p.brightness;
  document.getElementById('val-brightness').textContent = p.brightness;
  document.getElementById('master-contrast').value = p.contrast;
  document.getElementById('val-contrast').textContent = p.contrast;
}

async function applyMasterSettingsToAll() {
  const payload = {
    white_balance: {
      style: document.getElementById('master-wb-mode').value,
      gain_red: parseInt(document.getElementById('master-wb-red').value, 10),
      gain_blue: parseInt(document.getElementById('master-wb-blue').value, 10),
      gain_green: 50,
    },
    exposure: {
      mode: document.getElementById('master-exp-mode').value,
      shutter: document.getElementById('master-shutter').value,
      gain_max: parseInt(document.getElementById('master-gain-max').value, 10),
    },
    picture: {
      brightness: parseInt(document.getElementById('master-brightness').value, 10),
      contrast: parseInt(document.getElementById('master-contrast').value, 10),
    },
  };

  try {
    const res = await fetch('/api/cameras/batch-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`✅ Successfully synced parameters to all ${data.updated_cameras.length} cameras!`, 'success');
    } else {
      showToast(`Batch sync error: ${data.detail}`, 'error');
    }
  } catch (err) {
    showToast(`Network error syncing settings: ${err}`, 'error');
  }
}

async function saveSingleCameraTuning(cid) {
  const red = parseInt(document.getElementById(`tune-red-${cid}`).value, 10);
  const blue = parseInt(document.getElementById(`tune-blue-${cid}`).value, 10);
  const shutter = document.getElementById(`tune-shutter-${cid}`).value;

  try {
    const res = await fetch(`/api/cameras/${cid}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        wb: { style: 'Manual', gain_red: red, gain_blue: blue, gain_green: 50 },
        exp: { mode: 'Manual', shutter: shutter, gain_max: 50 },
      }),
    });
    if (res.ok) {
      showToast(`Saved tuning for ${cid.toUpperCase()}`, 'success');
    }
  } catch (err) {
    showToast(`Error saving ${cid}: ${err}`, 'error');
  }
}

// ==========================================================================
// Recording Lifecycle
// ==========================================================================

async function toggleRecording() {
  if (state.recording) {
    // STOP RECORDING
    try {
      const res = await fetch('/api/record/stop', { method: 'POST' });
      const data = await res.json();
      showToast(`Recording stopped. Manifest saved: ${data.manifest}`, 'success');
    } catch (err) {
      showToast(`Error stopping recording: ${err}`, 'error');
    }
  } else {
    // START RECORDING
    const sessionName = document.getElementById('session-name').value.trim();
    const durationInput = document.getElementById('duration-sec').value;
    const durationSec = durationInput ? parseFloat(durationInput) : null;
    const autoExtract = document.getElementById('auto-extract').checked;

    try {
      const res = await fetch('/api/record/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_name: sessionName || null,
          duration_sec: durationSec,
          auto_extract: autoExtract,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        showToast(`🔴 Recording started: ${data.session.run_id}`, 'warn');
      } else {
        showToast(`Failed to start recording: ${data.detail}`, 'error');
      }
    } catch (err) {
      showToast(`Error starting recording: ${err}`, 'error');
    }
  }
}

// ==========================================================================
// Diagnostics & Probe Modal
// ==========================================================================

async function triggerProbeAll() {
  const modal = document.getElementById('probe-modal');
  const tableContainer = document.getElementById('probe-results-table');
  modal.classList.remove('hidden');
  tableContainer.innerHTML = '<p class="text-muted">🔍 Probing camera streams and RTCP Sender Reports... please wait 4 seconds.</p>';

  try {
    const res = await fetch('/api/probe', { method: 'POST' });
    const data = await res.json();
    renderProbeTable(data.probe_results);
  } catch (err) {
    tableContainer.innerHTML = `<p class="diag-fail">Probe failed: ${err}</p>`;
  }
}

function renderProbeTable(results) {
  const tableContainer = document.getElementById('probe-results-table');
  if (!results || results.length === 0) {
    tableContainer.innerHTML = '<p class="text-muted">No probe results available.</p>';
    return;
  }

  let html = `
    <table class="diag-table">
      <thead>
        <tr>
          <th>Camera</th>
          <th>Status</th>
          <th>Codec</th>
          <th>Resolution</th>
          <th>Frames</th>
          <th>RTCP SRs (≥2)</th>
          <th>Sync Status</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const r of results) {
    const srsPass = (r.srs || 0) >= 2;
    html += `
      <tr>
        <td><strong>${r.camera_id.toUpperCase()}</strong></td>
        <td><span class="${r.ok ? 'diag-pass' : 'diag-fail'}">${r.ok ? 'ONLINE' : 'FAILED'}</span></td>
        <td>${r.codec || '—'}</td>
        <td>${r.resolution || '—'}</td>
        <td>${r.frames || 0}</td>
        <td>${r.srs || 0}</td>
        <td><span class="${srsPass ? 'diag-pass' : 'diag-fail'}">${srsPass ? 'READY' : 'LOW CONFIDENCE'}</span></td>
      </tr>
    `;
  }

  html += '</tbody></table>';
  tableContainer.innerHTML = html;
}

function closeProbeModal() {
  document.getElementById('probe-modal').classList.add('hidden');
}

// ==========================================================================
// Toast Notifications
// ==========================================================================

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
