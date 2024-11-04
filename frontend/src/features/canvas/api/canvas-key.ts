const canvasKey = {
  default: ['canvas'],
  blocks: (type: string) => [...canvasKey.default, 'blocks', type],
};

export default canvasKey;
