import { parseZipEntries } from './zip.js';
import { buildTriplets } from './organizer.js';
import { runPlaceholderMetrics } from './metrics.js';

self.addEventListener('message', async (event) => {
  const { id, type, payload } = event.data || {};

  if (type !== 'process-zip') {
    self.postMessage({ id, ok: false, error: 'Unknown worker message type' });
    return;
  }

  try {
    const entries = await parseZipEntries(payload.buffer);
    const triplets = buildTriplets(entries);
    const summary = runPlaceholderMetrics({ triplets, options: payload });

    self.postMessage({
      id,
      ok: true,
      payload: {
        summary,
        entriesCount: entries.length,
        tripletsCount: triplets.length,
      },
    });
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
});
