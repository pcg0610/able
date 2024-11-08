import { getOutgoers, type Node as XYFlowNode, type Edge as XYFlowEdge, type Connection } from '@xyflow/react';

// connection이 사이클 발생하는지 확인
export const isValidConnection = (nodes: XYFlowNode[], edges: XYFlowEdge[]) => {
  return (connection: Connection): boolean => {
    // 목적 노드(연결하려는 노드)
    const targetNode = nodes.find((node) => node.id === connection.target);
    // 목적 노드가 존재하지 않거나 출발 노드와 목적 노드가 동일하다면 사이클 발생 x
    if (!targetNode || targetNode.id === connection.source) return false;

    // 재귀적으로 자식 노드를 탐색하여 사이클을 감지
    const hasCycle = (node: XYFlowNode, visited = new Set<string>()): boolean => {
      // 이미 탐색한 노드이면 탐색 불필요
      if (visited.has(node.id)) return false;

      // 현재 노드 방문 처리
      visited.add(node.id);

      // 노드의 자식 노드를 추적하여 사이클 검출
      // 자식 노드 중 하나가 출발 노드와 동일하다면 사이클 발생
      // getOutgoers: 현재 노드에서 출발하는 모든 자식 노드(outgoers) 반환
      return getOutgoers(node, nodes, edges).some(
        (outgoer) => outgoer.id === connection.source || hasCycle(outgoer, visited)
      );
    };

    // 사이클이 존재하지 않으면 true 반환
    return !hasCycle(targetNode);
  };
};
