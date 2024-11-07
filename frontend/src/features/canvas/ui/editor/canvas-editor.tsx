import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  getOutgoers,
  type OnConnect,
  type Node as XYFlowNode,
  type IsValidConnection,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useCallback, useEffect, useState } from 'react';
import toast from 'react-hot-toast';

import * as S from '@features/canvas/ui/editor/canvas-editor.style';
import Common from '@shared/styles/common';
import {
  initialNodes,
  initialEdges,
} from '@features/canvas/model/initial-data';
import type { BlockItem } from '@features/canvas/types/block.type';
import {
  transformCanvasResponse,
  transformEdgesToEdgeSchema,
  transformNodesToBlockSchema,
} from '@features/canvas/utils/canvas-transformer.util';
import { useProjectStore } from '@/entities/project/model/project.model';
import { useFetchCanvas } from '@features/canvas/api/use-canvas.query';
import { useSaveCanvas } from '@features/canvas/api/use-canvas.mutation';
import { useNodeDropHandler } from '@features/canvas/model/use-node-drop-handler.model';
import { useNodeChangeHandler } from '@features/canvas/model/use-node-change-handler.modle';
import { useEdgeChangeHandler } from '@features/canvas/model/use-edge-change-handler.model';

import BlockNode from '@entities/block-node/block-node';
import BasicButton from '@shared/ui/button/basic-button';
import PlayIcon from '@icons/play.svg?react';
import SaveIcon from '@icons/save.svg?react';

const CanvasEditor = () => {
  const { currentProject } = useProjectStore();
  const { data } = useFetchCanvas(currentProject?.title || '');
  const { mutateAsync: saveCanvas } = useSaveCanvas();

  const [nodes, setNodes] = useNodesState(initialNodes);
  const [edges, setEdges] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<XYFlowNode | null>(null);

  const { handleNodesChange } = useNodeChangeHandler({
    nodes,
    setNodes,
    selectedNode,
    setSelectedNode,
  });
  const { handleEdgesChange } = useEdgeChangeHandler({
    edges,
    setEdges,
    nodes,
    selectedNode,
  });

  const { screenToFlowPosition, getNodes, getEdges } = useReactFlow();
  const { dropRef } = useNodeDropHandler({ setNodes, screenToFlowPosition });

  // 백엔드에서 캔버스 정보를 받아오면 노드와 엣지 상태를 업데이트
  useEffect(() => {
    if (data) {
      const transformedData = transformCanvasResponse(data);
      setNodes(transformedData.nodes);
      setEdges(transformedData.edges);
    }
  }, [data, setNodes, setEdges]);

  // 사이클 발생하는지 확인
  const isValidConnection: IsValidConnection = useCallback(
    (connection) => {
      const nodes = getNodes();
      const edges = getEdges();
      const target = nodes.find((node) => node.id === connection.target); // 연결하려는 노드
      const hasCycle = (node: XYFlowNode, visited = new Set()) => {
        // 이미 탐색한 노드이면 탐색 불필요
        if (visited.has(node.id)) return false;

        // 현재 노드 방문 처리
        visited.add(node.id);

        // 노드의 자식 노드를 추적하여 사이클 검출
        // getOutgoers: 현재 노드에서 출발하는 모든 자식 노드(outgoers) 반환
        for (const outgoer of getOutgoers(node, nodes, edges)) {
          // 자식 노드 중 하나가 출발 노드와 동일하다면 사이클 발생
          if (outgoer.id === connection.source) return true;
          if (hasCycle(outgoer, visited)) return true;
        }

        return false;
      };

      // 출발 노드와 목적 노드가 동일한지 확인
      if (target && target.id === connection.source) return false;
      return target ? !hasCycle(target) : false;
    },
    [getNodes, getEdges]
  );

  // 노드를 연결할 때 호출
  const onConnect: OnConnect = (connection) => {
    if (!isValidConnection(connection)) {
      toast.error('사이클 발생 위험이 있어요.');
      return;
    }

    // 사이클이 발생하지 않으면 엣지 추가
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
  };

  // 특정 노드의 블록 필드 변경
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

  const handleTrainButtonClick = () => {
    console.log('실행');
  };

  const handleSavaButtonClick = async () => {
    const transformedBlocks = transformNodesToBlockSchema(nodes);
    const transformedEdges = transformEdgesToEdgeSchema(edges);

    toast.promise(
      saveCanvas({
        projectName: currentProject?.title || '',
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
        onNodeClick={(_, node) => setSelectedNode(node)}
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
