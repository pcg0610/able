import styled from '@emotion/styled';

import Common from '@shared/styles/common';

export const HomeContentWrapper = styled.div`
  border-radius: 1rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 1.875rem;
`;

export const Title = styled.div`
  font-size: ${Common.fontSizes['3xl']};
  font-weight: ${Common.fontWeights.semiBold};
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

export const SubTitle = styled.h2`
  font-size: ${Common.fontSizes.xl};
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
  color: #a9a9a9;
`;

export const Description = styled.p`
  margin-top: 1rem;
  min-height: 1.25rem;
  color: ${Common.colors.gray400};
`;

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
  width: 18rem;
  height: 12rem;
  border-radius: 0.4rem;
  object-fit: cover;
  cursor: pointer;
`;

export const HistoryWrapper = styled.div`
  height: 100%;
  flex-direction: column;
  gap: 1rem;
`;
