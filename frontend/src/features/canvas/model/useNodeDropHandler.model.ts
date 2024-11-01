import { useDrop } from 'react-dnd';
import { Node } from '@xyflow/react';

import { useAddCenteredNode } from '@features/canvas/model/useAddCenteredNode.model';

interface DropHandlerParams {
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
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
    drop: (item: { type: string }, monitor) => {
      const clientOffset = monitor.getClientOffset();
      if (clientOffset) {
        addCenteredNode(clientOffset, item);
      }
    },
  }));

  return { dropRef };
};
