import * as S from '@features/home/ui/header/home-header.style';

import AbleLogo from '@assets/able-logo.png';

const HomeHeader = () => {
  return (
    <S.Header>
      <S.Logo src={AbleLogo} alt="logo"></S.Logo>
      <S.RightText>docs</S.RightText>
    </S.Header>
  );
};

export default HomeHeader;
