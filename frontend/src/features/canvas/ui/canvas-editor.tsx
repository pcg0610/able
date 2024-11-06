import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  type OnConnect,
  MarkerType,
  NodeChange,
  EdgeChange,
  Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useCallback, useEffect, useState } from 'react';

import * as S from '@features/canvas/ui/canvas-editor.style';
import Common from '@shared/styles/common';
import { useNodeDropHandler } from '@features/canvas/model/use-node-drop-handler.model';
import {
  initialNodes,
  initialEdges,
} from '@/features/canvas/model/initial-data';
import { useFetchCanvas } from '@/features/canvas/api/use-canvas.query';
import { useSaveCanvas } from '@features/canvas/api/use-canvas.mutation';
import type { BlockItem } from '@features/canvas/types/block.type';
import {
  transformCanvasResponse,
  transformEdgesToEdgeSchema,
  transformNodesToBlockSchema,
} from '@features/canvas/utils/canvas-transformer.util';

import BlockNode from '@entities/block-node/block-node';
import BasicButton from '@shared/ui/button/basic-button';
import PlayIcon from '@icons/play.svg?react';
import SaveIcon from '@icons/save.svg?react';
import toast from 'react-hot-toast';

const CanvasEditor = () => {
  const { data } = useFetchCanvas('춘식이');
  const { mutateAsync: saveCanvas } = useSaveCanvas();

  const [nodes, setNodes] = useNodesState(initialNodes);
  const [edges, setEdges] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  const { screenToFlowPosition } = useReactFlow();
  const { dropRef } = useNodeDropHandler({ setNodes, screenToFlowPosition });

  useEffect(() => {
    if (data) {
      const transformedData = transformCanvasResponse(data);
      setNodes(transformedData.nodes);
      setEdges(transformedData.edges);
    }
  }, [data, setNodes, setEdges]);

  const onConnect: OnConnect = (connection) =>
    setEdges((eds) =>
      addEdge(
        {
          ...connection,
          type: 'smoothstep',
          markerEnd: { type: MarkerType.ArrowClosed, width: 30, height: 30 },
        },
        eds
      )
    );

  const handleFieldChange = useCallback(
    (nodeId: string, fieldName: string, value: string) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  block: {
                    ...(node.data.block as BlockItem),
                    fields: (node.data.block as BlockItem).fields.map((field) =>
                      field.name === fieldName ? { ...field, value } : field
                    ),
                  },
                },
              }
            : node
        )
      );
    },
    [setNodes]
  );

  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes((nds) => {
        const filteredChanges = changes.filter((change) => {
          if (change.type === 'remove' && 'id' in change) {
            // id 속성을 가진 경우에만 접근
            const node = nds.find((node) => node.id === change.id);
            const data = node?.data as { block: BlockItem };
            return data?.block?.type !== 'data';
          }
          return true;
        });

        return applyNodeChanges(filteredChanges, nds);
      });

      // 노드가 변경될 때마다 선택된 노드를 업데이트
      const selectedChange = changes.find(
        (change) =>
          change.type === 'select' && 'id' in change && change.selected
      );
      if (selectedChange && 'id' in selectedChange) {
        const selectedNode =
          nodes.find((node) => node.id === selectedChange.id) || null;
        setSelectedNode(selectedNode);
      }
    },
    [setNodes, nodes]
  );

  // 엣지 삭제 제어
  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges((eds) => {
        const filteredChanges = changes.filter((change) => {
          if (change.type === 'remove' && selectedNode) {
            // 현재 선택된 노드가 data 타입일 경우 해당 노드와 연결된 엣지는 삭제하지 않음
            const sourceNode = nodes.find(
              (node) => node.id === selectedNode.id
            );
            const data = sourceNode?.data as { block: BlockItem };
            if (data?.block?.type === 'data') {
              const edge = eds.find((edge) => edge.id === change.id);
              if (
                edge &&
                (edge.source === selectedNode.id ||
                  edge.target === selectedNode.id)
              ) {
                return false;
              }
            }
          }
          return true;
        });

        return applyEdgeChanges(filteredChanges, eds);
      });
    },
    [setEdges, nodes, selectedNode]
  );

  const handleTrainButtonClick = () => {
    console.log('실행');
  };

  const handleSavaButtonClick = async () => {
    const transformedBlocks = transformNodesToBlockSchema(nodes);
    const transformedEdges = transformEdgesToEdgeSchema(edges);

    toast.promise(
      saveCanvas({
        projectName: '춘식이',
        canvas: { blocks: transformedBlocks, edges: transformedEdges },
      }),
      {
        loading: '저장 중...',
        success: '저장 완료',
        error: '저장 실패',
      }
    );
  };

  return (
    <S.Canvas ref={dropRef}>
      <ReactFlow
        nodes={nodes.map((node) => ({
          ...node,
          data: {
            ...node.data,
            onFieldChange: (fieldName: string, value: string) =>
              handleFieldChange(node.id, fieldName, value),
          },
        }))}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={onConnect}
        onNodeClick={(_, node) => setSelectedNode(node)} // 노드를 클릭할 때 선택된 노드 업데이트
        nodeTypes={{ custom: BlockNode }}
      >
        <Controls position='bottom-center' orientation='horizontal' />
        <Background variant={BackgroundVariant.Dots} />
      </ReactFlow>
      <S.OverlayButton>
        <BasicButton
          text='실행'
          icon={<PlayIcon width={13} height={16} />}
          width='5.5rem'
          onClick={handleTrainButtonClick}
        />
        <BasicButton
          text='저장'
          color={Common.colors.primary}
          backgroundColor={Common.colors.secondary}
          icon={<SaveIcon />}
          width='5.5rem'
          onClick={handleSavaButtonClick}
        />
      </S.OverlayButton>
    </S.Canvas>
  );
};

export default CanvasEditor;
