import { useDrop } from 'react-dnd';
import { Node as XYFlowNode } from '@xyflow/react';
import { Dispatch, SetStateAction } from 'react';

import { BlockItem } from '@features/canvas/types/block.type';
import { useAddCenteredNode } from '@features/canvas/model/use-add-centered-node.model';

interface DropHandlerParams {
  setNodes: Dispatch<SetStateAction<XYFlowNode[]>>;
  screenToFlowPosition: (position: { x: number; y: number }) => {
    x: number;
    y: number;
  };
}

export const useNodeDropHandler = ({
  setNodes,
  screenToFlowPosition,
}: DropHandlerParams) => {
  const addCenteredNode = useAddCenteredNode({
    setNodes,
    screenToFlowPosition,
  });

  const [, dropRef] = useDrop(() => ({
    accept: 'BLOCK',
    drop: (item: BlockItem, monitor) => {
      const clientOffset = monitor.getClientOffset();
      if (clientOffset) {
        addCenteredNode(clientOffset, item);
      }
    },
  }));

  return { dropRef };
};
