export const INTENT_TYPES = {
  FLOW: 'FLOW',
  DEPENDENCY: 'DEPENDENCY',
  IMPACT: 'IMPACT',
  EXPLAINER: 'EXPLAINER'
};

export function detectQueryIntent(query) {
  const q = query.toLowerCase();
  
  if (q.includes('break') || q.includes('impact') || q.includes('change') || q.includes('modify')) {
    return INTENT_TYPES.IMPACT;
  }
  if (q.includes('depend') || q.includes('who calls') || q.includes('calls') || q.includes('imports')) {
    return INTENT_TYPES.DEPENDENCY;
  }
  if (q.includes('explain') || q.includes('summary') || q.includes('what does this function')) {
    return INTENT_TYPES.EXPLAINER;
  }
  
  // Default intent for "how does X work" or request traces
  return INTENT_TYPES.FLOW;
}