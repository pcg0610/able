import { Link } from 'react-router-dom';

import * as S from '@pages/home.style'
import HomeHeader from '@features/home/ui/header/home-header'
import HomeSideBar from '@features/home/ui/sidebar/home-sidebar'

const HomePage = () => {
  return (
    <>
      <HomeHeader />
      <S.PageLayout>
        <HomeSideBar />
        <S.ContentContainer>
          <div>홈화면</div>
          <Link to={'/canvas'}>캔버스</Link>
          &nbsp;
          <Link to={'/train'}>학습</Link>
        </S.ContentContainer>
      </S.PageLayout>
    </>
  );
};

export default HomePage;
