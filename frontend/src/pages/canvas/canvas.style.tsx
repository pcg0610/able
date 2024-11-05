import styled from '@emotion/styled';

export const PageContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
`;

export const Content = styled.div`
  display: flex;
  flex-grow: 1;
  overflow: hidden;
  position: relative;
`;

export const OverlayButton = styled.div`
  display: flex;
  position: absolute;
  gap: 0.6rem;

  top: 1rem;
  right: 1rem;
  z-index: 10;
`;
