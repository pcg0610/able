import { css } from '@emotion/react';

import Common from '@shared/styles/common';

export const globalStyle = css`
  @font-face {
    font-family: 'Pretendard';
    src: url('/src/assets/fonts/PretendardVariable.woff2')
      format('woff2-variations');
    font-weight: 400 600;
    font-style: normal;
  }

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Pretendard', sans-serif;
    color: ${Common.colors.black};
  }
`;
