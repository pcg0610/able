import TransformIcon from '@assets/icons/transform.svg?react';
import LayerIcon from '@assets/icons/layer.svg?react';
import ActivationIcon from '@assets/icons/activation.svg?react';
import LossIcon from '@assets/icons/loss.svg?react';
import OperationIcon from '@assets/icons/operation.svg?react';
import OptimizerIcon from '@assets/icons/optimizer.svg?react';
import ModuleIcon from '@assets/icons/module.svg?react';

export const BLOCK_MENU = [
  {
    name: 'transform',
    icon: TransformIcon,
    color: '#FF686B',
  },
  {
    name: 'layer',
    icon: LayerIcon,
    color: '#34D399',
  },
  {
    name: 'activation',
    icon: ActivationIcon,
    color: '#71D334',
  },
  {
    name: 'loss',
    icon: LossIcon,
    color: '#EE74B5',
  },
  {
    name: 'operation',
    icon: OperationIcon,
    color: '#A768FF',
  },
  {
    name: 'optimizer',
    icon: OptimizerIcon,
    color: '#FFD00D',
  },
  {
    name: 'module',
    icon: ModuleIcon,
    color: '#30AEF7',
  },
] as const;
