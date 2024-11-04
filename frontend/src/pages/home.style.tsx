import styled from '@emotion/styled';

export const PageLayout = styled.div`
  display: flex;
  height: calc(100vh - 2.5rem); 
`;

export const ContentContainer = styled.div`
  flex: 1;
  padding: 1.25rem; 
  overflow-y: auto; 
  height: calc(100% - 2.5rem);
  box-sizing: border-box;

  &::-webkit-scrollbar {
    width: 0.9375rem;
    height: 0.5rem;
    border-radius: 0.375rem;
    background: rgba(255, 255, 255, 0.4);
  }
  &::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 0.375rem;
  }
`;