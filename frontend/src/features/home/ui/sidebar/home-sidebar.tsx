import * as S from '@features/home/ui/sidebar/home-sidebar.style';

const HomeSideBar = () => {

   return (
      <S.SidebarContainer>
         <S.Title>내 프로젝트</S.Title>
         <S.Subtitle>내가 생성한 모델 모아보기</S.Subtitle>
         <S.Button>프로젝트 만들기</S.Button>

         <S.FolderSection>
            <S.Folder>
               <S.Icon>📁</S.Icon> 컴퓨터 비전 프로젝트
            </S.Folder>
            <S.Folder>
               <S.Icon>📂</S.Icon> 홍박사팀
            </S.Folder>
            <S.Folder>
               <S.Icon>📂</S.Icon> 박찬규 연구실
            </S.Folder>
         </S.FolderSection>

         <S.Footer>
            <S.FooterIcon>🚀</S.FooterIcon>
            <div>
               <S.FooterText>서버 확인하기</S.FooterText>
               <S.FooterStatus>Running...</S.FooterStatus>
            </div>
         </S.Footer>
      </S.SidebarContainer>
   );
};

export default HomeSideBar;
