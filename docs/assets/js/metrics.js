// Placeholder de calculo.
// Aqui se migraran funciones de src/Single.py en siguientes iteraciones.

export function runPlaceholderMetrics({ triplets, options }) {
  return {
    mode: 'browser-local-placeholder',
    applyFilter: options.applyFilter,
    cutoffMm: options.cutoffMm,
    specimensDetected: triplets.length,
    note: 'Estructura base creada. Pendiente port de formulas ISO 4287/13565-2 a JS.',
  };
}
