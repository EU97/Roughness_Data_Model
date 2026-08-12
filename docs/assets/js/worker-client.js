export function createWorkerClient(workerUrl) {
  const worker = new Worker(workerUrl, { type: 'module' });

  function request(type, payload) {
    return new Promise((resolve, reject) => {
      const id = crypto.randomUUID();

      function onMessage(event) {
        const msg = event.data;
        if (!msg || msg.id !== id) {
          return;
        }

        worker.removeEventListener('message', onMessage);
        if (msg.ok) {
          resolve(msg.payload);
        } else {
          reject(new Error(msg.error || 'Worker error'));
        }
      }

      worker.addEventListener('message', onMessage);
      worker.postMessage({ id, type, payload });
    });
  }

  return {
    processZip(payload) {
      return request('process-zip', payload);
    },
  };
}
