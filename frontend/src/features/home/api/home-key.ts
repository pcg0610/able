const homeKey = {
  default: ['home'],
  list: () => [...homeKey.default, 'list'],
  project: (title: string) => [...homeKey.default, 'project', title],
};

export default homeKey;
