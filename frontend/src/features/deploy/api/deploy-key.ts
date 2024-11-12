const deployKey = {
  default: ['train'],
  regist: (projectName: string, trainResult: string, checkpoint: string, uri: string, descriptioon: string) => [
    ...deployKey.default,
    'regist',
    projectName,
    trainResult,
    checkpoint,
    uri,
    descriptioon,
  ],
};

export default deployKey;
