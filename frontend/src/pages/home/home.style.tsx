import styled from '@emotion/styled';

export const PageLayout = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
`;

export const PageContainer = styled.div`
  display: flex;
  height: calc(100vh - 2.5rem);
`;

export const ContentContainer = styled.div`
  flex: 1;
  padding: 1.875rem;
  overflow-y: auto;
  box-sizing: border-box;
`;
