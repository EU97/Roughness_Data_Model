function getStem(fileName) {
  const lower = fileName.toLowerCase();
  if (lower.endsWith('.tx1') || lower.endsWith('.tx2') || lower.endsWith('.tx3')) {
    return fileName.slice(0, -4);
  }
  return null;
}

export function buildTriplets(entries) {
  const map = new Map();

  for (const entry of entries) {
    const stem = getStem(entry.name);
    if (!stem) {
      continue;
    }

    const ext = entry.name.slice(-4).toLowerCase();
    if (!map.has(stem)) {
      map.set(stem, {});
    }

    map.get(stem)[ext] = entry;
  }

  const complete = [];
  for (const [stem, parts] of map.entries()) {
    if (parts['.tx1'] && parts['.tx2'] && parts['.tx3']) {
      complete.push({ stem, files: parts });
    }
  }

  return complete;
}
