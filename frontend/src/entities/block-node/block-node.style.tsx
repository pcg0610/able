// src/entities/canvas/block-node.style.ts
import { css } from '@emotion/react';

// 블록 유형과 색상 매핑
export const blockColors: Record<string, string> = {
  transform: '#FF6347',
  layer: '#66CDAA',
  activation: '#4682B4',
  loss: '#FFD700',
  operation: '#FF8C00',
  optimizer: '#6A5ACD',
  model: '#8A2BE2',
};

// BlockNode 컨테이너 스타일
export const containerStyle = (blockColor: string) => css`
  background-color: ${blockColor};
  padding: 10px;
  border-radius: 8px;
  width: 150px;
`;

// Label 스타일
export const labelStyle = css`
  font-weight: bold;
  text-align: center;
`;

// Field 스타일
export const fieldStyle = css`
  margin-top: 10px;
`;

// Input Wrapper 스타일
export const inputWrapperStyle = css`
  margin-bottom: 8px;
`;

// Input 스타일
export const inputStyle = css`
  width: 100%;
  margin-top: 4px;
`;
