import styled from '@emotion/styled';

import Common from '@/shared/styles/common';

export const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 1rem;
  border-bottom: 0.0625rem solid #ddd;
  height: 2.5rem;
`;

export const LeftText = styled.div`
  font-size: ${Common.fontSizes.lg};
  font-weight: bold;
  color: ${Common.colors.primary};
`;

export const RightText = styled.div`
  font-size: ${Common.fontSizes.base};
  color: ${Common.colors.gray500};
`;
