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
  list: (page: number, pageSize: number) => [...deployKey.default, 'list', page, pageSize],
  stop: (uri: string) => [...deployKey.default, 'stop', uri],
};

export default deployKey;
