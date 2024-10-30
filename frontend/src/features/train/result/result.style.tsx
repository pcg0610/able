import styled from '@emotion/styled';

export const GraphContainer = styled.div`
  padding: 20px;
  border-radius: 8px;
  background: #f9f9f9;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

export const GraphTitle = styled.h3`
  font-size: 1.2rem;
  font-weight: 500;
  color: #333;
`;

export const LegendContainer = styled.div`
  display: flex;
  gap: 10px;
  margin-top: 10px;
  margin-bottom: 20px;
`;

export const LegendItem = styled.div`
  display: flex;
  align-items: center;
  font-size: 0.9rem;
  color: #666;
`;

export const BlueDot = styled.span`
  width: 10px;
  height: 10px;
  background-color: #1f77b4;
  border-radius: 50%;
  display: inline-block;
  margin-right: 5px;
`;

export const BlueLine = styled.span`
  width: 20px;
  height: 2px;
  background-color: #1f77b4;
  display: inline-block;
  margin-right: 5px;
`;

export const GraphPlaceholder = styled.div`
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  color: #999;
  border: 1px dashed #ddd;
  border-radius: 8px;
  margin-top: 20px;
`;