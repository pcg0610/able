import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

import * as S from '@features/home/ui/sidebar/home-sidebar.style';
import Common from '@shared/styles/common';
import { useProjects } from '@features/home/api/use-home.query';
import { useProjectNameStore } from '@entities/project/model/project.model';

import ProjectModal from '@/features/home/ui/modal/project-modal';
import BasicButton from '@shared/ui/button/basic-button';
import FileIcon from '@icons/file.svg?react';
import RocketIcon from '@icons/rocket.svg?react';
import LoadingSpinner from '@icons/horizontal-loading.svg?react';

const HomeSideBar = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const navigate = useNavigate();

  const { projectName, setProjectName } = useProjectNameStore();
  const { data: projects, isLoading } = useProjects();

  useEffect(() => {
    if (projects && projects.length > 0 && !projectName) {
      setProjectName(projects[0]);
    }
  }, [projects, projectName, setProjectName]);

  const handleClick = (project: string) => {
    setProjectName(project);
  };

  const handleServer = () => {
    navigate('/deploy');
  };

  const handleCreateProjectClick = () => {
    setIsModalOpen(true);
    setIsClosing(false);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const handleAnimationEnd = () => {
    if (isClosing) {
      setIsModalOpen(false);
      setIsClosing(false);
    }
  };

  return (
    <S.SidebarContainer>
      <S.Title>내 프로젝트</S.Title>
      <S.Subtitle>내가 생성한 모델 모아보기</S.Subtitle>
      <BasicButton
        color={Common.colors.primary}
        backgroundColor={Common.colors.secondary}
        text="프로젝트 만들기"
        width="100%"
        onClick={() => {
          handleCreateProjectClick();
        }}
      />

      <S.FolderSection>
        {isLoading ? (
          <LoadingSpinner height={50} />
        ) : (
          projects?.map((project: string, index: number) => (
            <S.Folder key={index} isSelected={projectName === project} onClick={() => handleClick(project)}>
              <FileIcon width={20} height={20} /> {project}
            </S.Folder>
          ))
        )}
      </S.FolderSection>

      <S.Footer onClick={handleServer}>
        <S.RocketCircle>
          <RocketIcon width={40} height={40} />
        </S.RocketCircle>
        <div>
          <S.FooterText>서버 확인하기</S.FooterText>
          <S.FooterStatus>Running...</S.FooterStatus>
        </div>
      </S.Footer>
      {isModalOpen && (
        <ProjectModal onClose={closeModal} isClosing={isClosing} onAnimationEnd={handleAnimationEnd} type={'create'} />
      )}
    </S.SidebarContainer>
  );
};

export default HomeSideBar;
