import TransformIcon from '@assets/icons/transform.svg?react';
import LayerIcon from '@assets/icons/layer.svg?react';
import ActivationIcon from '@assets/icons/activation.svg?react';
import LossIcon from '@assets/icons/loss.svg?react';
import OperationIcon from '@assets/icons/operation.svg?react';
import OptimizerIcon from '@assets/icons/optimizer.svg?react';
import ModuleIcon from '@assets/icons/module.svg?react';

export const MENU_ICON_MAP = {
  TransformIcon,
  LayerIcon,
  ActivationIcon,
  LossIcon,
  OperationIcon,
  OptimizerIcon,
  ModuleIcon,
};

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
