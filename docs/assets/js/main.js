import { createWorkerClient } from './worker-client.js';

const zipInput = document.getElementById('zipInput');
const applyFilterEl = document.getElementById('applyFilter');
const cutoffEl = document.getElementById('cutoff');
const runBtn = document.getElementById('runBtn');
const downloadBtn = document.getElementById('downloadBtn');
const statusEl = document.getElementById('status');
const summaryEl = document.getElementById('summary');

let lastOutput = null;
const workerClient = createWorkerClient(new URL('./worker.js', import.meta.url));

function setStatus(text) {
  statusEl.textContent = text;
}

function setSummary(payload) {
  summaryEl.textContent = JSON.stringify(payload, null, 2);
}

async function runProcessing() {
  const file = zipInput.files?.[0];
  if (!file) {
    setStatus('Selecciona un archivo ZIP antes de procesar.');
    return;
  }

  runBtn.disabled = true;
  downloadBtn.disabled = true;
  setStatus('Leyendo ZIP en el navegador...');

  try {
    const arrayBuffer = await file.arrayBuffer();
    setStatus('Ejecutando pipeline local en Web Worker...');

    const result = await workerClient.processZip({
      fileName: file.name,
      buffer: arrayBuffer,
      applyFilter: Boolean(applyFilterEl.checked),
      cutoffMm: Number(cutoffEl.value || 0.8),
    });

    lastOutput = result;
    setSummary(result.summary);
    setStatus('Proceso finalizado en tu dispositivo.');
    downloadBtn.disabled = false;
  } catch (err) {
    setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    runBtn.disabled = false;
  }
}

function downloadJson() {
  if (!lastOutput) {
    return;
  }

  const blob = new Blob([JSON.stringify(lastOutput, null, 2)], {
    type: 'application/json',
  });

  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'roughness_web_result.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

runBtn.addEventListener('click', runProcessing);
downloadBtn.addEventListener('click', downloadJson);
