import { Dispatch, SetStateAction, useCallback } from 'react';
import { Node } from '@xyflow/react';

interface ClientOffset {
  x: number;
  y: number;
}

interface AddNodeParams {
  setNodes: Dispatch<SetStateAction<Node[]>>;
  screenToFlowPosition: (position: { x: number; y: number }) => {
    x: number;
    y: number;
  };
}

export const useAddCenteredNode = ({
  setNodes,
  screenToFlowPosition,
}: AddNodeParams) => {
  // 노드의 중앙 위치 조정 로직을 useCallback으로 작성하여 의존성 문제 해결
  const adjustNodePosition = useCallback(
    (nodeId: string, clientOffset: ClientOffset) => {
      setTimeout(() => {
        const element = document.querySelector(`[data-id="${nodeId}"]`);
        if (element) {
          const { width, height } = element.getBoundingClientRect();
          const centeredPosition = screenToFlowPosition({
            x: clientOffset.x - width / 2,
            y: clientOffset.y - height / 2,
          });

          setNodes((nds) =>
            nds.map((node) =>
              node.id === nodeId
                ? { ...node, position: centeredPosition }
                : node
            )
          );
        }
      }, 0);
    },
    [screenToFlowPosition, setNodes]
  );

  // 노드를 추가하고 adjustNodePosition 호출
  return useCallback(
    (clientOffset: ClientOffset, item: { type: string }) => {
      const initialPosition = screenToFlowPosition({
        x: clientOffset.x,
        y: clientOffset.y,
      });

      const newNode: Node = {
        id: `${Math.random()}`,
        type: 'custom',
        position: initialPosition,
        data: {
          type: item.type,
          fields: [
            { name: 'in_channels', required: true },
            { name: 'out_channels', required: true },
            { name: 'kernel_size', required: true },
          ],
        },
      };

      setNodes((nds) => [...nds, newNode]);

      // 중앙 위치 조정 후 갱신
      adjustNodePosition(newNode.id, clientOffset);
    },
    [screenToFlowPosition, setNodes, adjustNodePosition]
  );
};
