import styled from '@emotion/styled';
import { Common } from '@shared/styles/common';

export const Container = styled.div`
  padding: 0 1.25rem;
  height: 100vh;
  display: flex;
  flex-direction: column;
`;

export const Header = styled.div`
  display: flex;
  justify-content: end;
  align-items: center;
  margin-bottom: 1.25rem;
  flex-shrink: 0;
`;

export const GridContainer = styled.div`
  display: grid;
  gap: 1.25rem;
  grid-template-rows: 1.1fr 0.9fr;
  flex-grow: 1;
  height: 100%;

  & > .top-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5625rem;
  }

  & > .bottom-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
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
  align-items: center;
`;

export const GraphTitle = styled.h3`
  margin-top: .3125rem;
  margin-left: .625rem;
  margin-right: auto;
  font-size: 1.25rem;
  color: #333;
`;