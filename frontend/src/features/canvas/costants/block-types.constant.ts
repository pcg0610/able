export const BLOCK_MENU = [
  {
    name: 'transform',
    icon: 'TransformIcon', // 아이콘 컴포넌트 이름
    color: '#FF5733', // 임시 블록 색상
  },
  {
    name: 'layer',
    icon: 'LayerIcon',
    color: '#33C1FF',
  },
  {
    name: 'activation',
    icon: 'ActivationIcon',
    color: '#75FF33',
  },
  {
    name: 'loss',
    icon: 'LossIcon',
    color: '#FF33A1',
  },
  {
    name: 'operation',
    icon: 'OperationIcon',
    color: '#9B33FF',
  },
  {
    name: 'optimizer',
    icon: 'OptimizerIcon',
    color: '#FFD733',
  },
  {
    name: 'module',
    icon: 'ModuleIcon',
    color: '#33FFAA',
  },
] as const;

export type MenuName = (typeof BLOCK_MENU)[number]['name'];
