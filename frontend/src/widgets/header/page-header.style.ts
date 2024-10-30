import styled from '@emotion/styled';

import Common from '@/shared/styles/common';

export const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: ${Common.colors.primary};
  color: ${Common.colors.white};
  padding: 0 1rem;
  height: 2.5rem;
`;

export const Title = styled.span`
  font-weight: ${Common.fontWeights.semiBold};
`;

export const Date = styled.span`
  font-size: ${Common.fontSizes.xs};
  margin-left: 0.625rem;
`;
