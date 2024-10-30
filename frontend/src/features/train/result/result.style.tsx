import styled from '@emotion/styled';

export const GraphContainer = styled.div`
  padding: 1.25rem;
  border-radius: .5rem;
  background: #f9f9f9;
  box-shadow: 0 .125rem .25rem rgba(0, 0, 0, 0.1);
`;

export const GraphTitle = styled.h3`
  font-size: 1.2px;
  font-weight: 500;
  color: #333;
`;

export const LegendContainer = styled.div`
  display: flex;
  gap: .625rem;
  margin-top: .625rem;
  margin-bottom: 1.25rem;
`;

export const LegendItem = styled.div`
  display: flex;
  align-items: center;
  font-size: 0.9rem;
  color: #666;
`;

export const BlueDot = styled.span`
  width: .625rem;
  height: .625rem;
  background-color: #1f77b4;
  border-radius: 50%;
  display: inline-block;
  margin-right: .3125rem;
`;

export const BlueLine = styled.span`
  width: 1.25rem;
  height: .125rem;
  background-color: #1f77b4;
  display: inline-block;
  margin-right: .3125rem;
`;

export const GraphPlaceholder = styled.div`
  height: 12.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: #999;
  border: .0625rem dashed #ddd;
  border-radius: .5rem;
  margin-top: 1.25rem;
`;