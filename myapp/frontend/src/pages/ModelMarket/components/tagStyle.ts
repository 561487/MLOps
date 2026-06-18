/** Unified tag color mapping — shared between list cards and detail page */

export const getModelTagColor = (tag?: string) => {
  const value = (tag || '').toLowerCase();

  if (value.includes('目标检测') || value.includes('object')) return 'green';
  if (value.includes('yolo')) return 'cyan';
  if (value.includes('pytorch') || value.includes('torch')) return 'blue';
  if (value.includes('视觉') || value.includes('vision')) return 'cyan';
  if (value.includes('online')) return 'green';
  if (value.includes('tensorflow')) return 'orange';
  if (value.includes('transformer') || value.includes('llm') || value.includes('大模型')) return 'purple';

  return 'default';
};
