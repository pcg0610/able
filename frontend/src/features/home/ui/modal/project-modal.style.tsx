import styled from '@emotion/styled';

import Common from '@shared/styles/common';

export const DropDownWrapper = styled.div`
  display: flex;
  flex-direction: column;
`;

export const Label = styled.label`
  font-size: ${Common.fontSizes.sm};
  margin-bottom: 0.25rem;
`;

export const Select = styled.select`
  padding: 0.5rem;
  border: 0.0625rem solid #ddd;
  border-radius: 0.25rem;
  font-size: ${Common.fontSizes.sm};
`;
