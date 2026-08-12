// Placeholder de parser ZIP.
// En siguiente iteracion se integrara una libreria (por ejemplo fflate o JSZip)
// para extraer archivos reales desde el ArrayBuffer.

export async function parseZipEntries(_arrayBuffer) {
  return [
    { path: 'mock/S1.tx1', name: 'S1.tx1' },
    { path: 'mock/S1.tx2', name: 'S1.tx2' },
    { path: 'mock/S1.tx3', name: 'S1.tx3' },
    { path: 'mock/S2.tx1', name: 'S2.tx1' },
    { path: 'mock/S2.tx2', name: 'S2.tx2' },
    { path: 'mock/S2.tx3', name: 'S2.tx3' },
  ];
}
