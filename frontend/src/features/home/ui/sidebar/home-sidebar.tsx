import { useState } from 'react';

import * as S from '@features/home/ui/sidebar/home-sidebar.style';
import Common from '@shared/styles/common';

import BasicButton from '@shared/ui/button/basic-button'
import FileIcon from '@icons/file.svg?react';
import RocketIcon from '@icons/rocket.svg?react';

const HomeSideBar = () => {
   const [selectedProject, setSelectedProject] = useState("홍박사팀");

   const projects = [
      "컴퓨터 비전 프로젝트",
      "홍박사팀",
      "박찬규 연구실",
   ];

   const handleClick = (project: string) => {
      setSelectedProject(project);
   };

   const handleServer = () => {
      console.log("router server");
   };

   return (
      <S.SidebarContainer>
         <S.Title>내 프로젝트</S.Title>
         <S.Subtitle>내가 생성한 모델 모아보기</S.Subtitle>
         <BasicButton
            color={Common.colors.primary}
            backgroundColor={Common.colors.secondary}
            text="프로젝트 만들기"
            width='100%'
            onClick={() => {
               console.log('모델 실행 버튼 클릭됨');
            }}
         />

         <S.FolderSection>
            {projects.map((project, index) => (
               <S.Folder
                  key={index}
                  isSelected={selectedProject === project}
                  onClick={() => handleClick(project)}>
                  <FileIcon width={20} height={20} /> {project}
               </S.Folder>
            ))}
         </S.FolderSection>

         <S.Footer onClick={handleServer}>
            <RocketIcon width={59} height={59} />
            <div>
               <S.FooterText>서버 확인하기</S.FooterText>
               <S.FooterStatus>Running...</S.FooterStatus>
            </div>
         </S.Footer>
      </S.SidebarContainer>
   );
};

export default HomeSideBar;
