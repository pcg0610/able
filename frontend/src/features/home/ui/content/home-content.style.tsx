import styled from '@emotion/styled';

import Common from '@shared/styles/common';

export const HomeContentWrapper = styled.div`
  padding: 0.625rem 2rem;
  border-radius: 1rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 1.875rem;
`;

export const Title = styled.div`
  font-size: ${Common.fontSizes['3xl']};
  font-weight: bold;
  display: flex;
  align-items: end;
  gap: 0.75rem;
`;

export const SubTitle = styled.h2`
  font-size: ${Common.fontSizes['2xl']};
  font-weight: ${Common.fontWeights.medium};
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: ${Common.colors.gray500};
  margin-bottom: 0.9375rem;
`;

export const PythonTitle = styled.span`
  font-size: 14px;
  font-weight: ${Common.fontWeights.regular};
  color: #A9A9A9;
`

export const CanvasWrapper = styled.div`
  display: flex;
  justify-content: center; 
  align-items: center;
  background-color: #fff;
  padding: 1rem;
  border-radius: 0.5rem;
  box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
`;

export const CanvasImage = styled.img`
  max-width: 21.875rem;
  height: auto;
  border-radius: 0.5rem;
  object-fit: cover;
  cursor: pointer;
`;

export const HistoryWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem; 
`;
