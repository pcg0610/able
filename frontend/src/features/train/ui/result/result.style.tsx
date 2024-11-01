import styled from '@emotion/styled';
import Common from '@shared/styles/common';

export const Container = styled.div`
  padding: 0.7rem 5.5rem 1.7rem;
  height: calc(100vh - 2.5rem);
  display: flex;
  flex-direction: column;
  background-color: ${Common.colors.background};
`;

export const Header = styled.div`
  display: flex;
  justify-content: end;
  align-items: center;
  margin-bottom: 0.7rem;
  flex-shrink: 0;
`;

export const GridContainer = styled.div`
  display: grid;
  gap: 1.25rem;
  grid-template-rows: 1.1fr 1fr;
  flex-grow: 1;
  height: 100%;

  & > .top-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.6rem;
  }

  & > .bottom-row {
    display: grid;
    grid-template-columns: 1.1fr 0.7fr 1.2fr;
    gap: 1.6rem;
  }
`;

export const GraphCard = styled.div`
  background: ${Common.colors.white};
  padding: 1.25rem;
  border-radius: .5rem;
  box-shadow: 0 .0625rem .25rem rgba(0, 0, 0, 0.1);
  border: .0625rem solid ${Common.colors.gray200};
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
`;

export const GraphTitle = styled.h3`
  margin: .2rem 0 .525rem .625rem;
  margin-right: auto;
  font-size: ${Common.fontSizes.xl};
  font-weight: ${Common.fontWeights.medium};
  color: #333;
`;

export const F1ScoreTitle = styled(GraphTitle)`
  font-weight: ${Common.fontWeights.semiBold}; 
  margin-left: auto;
  font-size: ${Common.fontSizes['3xl']};
`;