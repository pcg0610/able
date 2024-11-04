import * as S from '@pages/home.style';

import HomeHeader from '@features/home/ui/header/home-header';
import HomeSideBar from '@features/home/ui/sidebar/home-sidebar';
import HomeContent from '@/features/home/ui/content/home-content';

const HomePage = () => {
  return (
    <>
      <HomeHeader />
      <S.PageLayout>
        <HomeSideBar />
        <S.ContentContainer>
          <HomeContent />
        </S.ContentContainer>
      </S.PageLayout>
    </>
  );
};

export default HomePage;
