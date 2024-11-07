import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

import * as S from '@/features/home/ui/content/home-content.style';
import Common from '@shared/styles/common';
import { useProject } from '@features/home/api/use-home.query';
import { useProjectStore, useProjectStateStore } from '@entities/project/model/project.model';

import HistoryList from '@/features/home/ui/content/history-list';
import Pagination from '@shared/ui/pagination/pagination';
import ProjectModal from '@shared/ui/modal/project-modal';
import WritingIcon from '@icons/writing.svg?react';
import ClockIcon from '@icons/clock.svg?react';
import FolderIcon from '@icons/folder.svg?react';
import SettingIcon from '@icons/setting.svg?react';

const HomeContent = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const historyItems = [
    { id: 1, date: '2024.10.23 16:35', accuracy: '73%', status: '완료' },
    { id: 2, date: '2024.10.23 16:40', accuracy: '85%', status: '진행 중' },
    { id: 3, date: '2024.10.23 16:40', accuracy: '85%', status: '진행 중' },
    { id: 4, date: '2024.10.23 16:40', accuracy: '85%', status: '진행 중' },
    { id: 5, date: '2024.10.23 16:40', accuracy: '85%', status: '진행 중' },
  ];

  const navigate = useNavigate();
  const [currentPage, setCurrentPage] = useState(1);

  const { projectName } = useProjectStateStore();
  const { currentProject, setCurrentProject } = useProjectStore();

  const { data: project, isLoading, error } = useProject(projectName, {
    enabled: !!projectName,
  });

  const handleCanvasClick = () => {
    navigate('/canvas');
  };

  const handleSettingClick = () => {
    setIsModalOpen(true);
    setIsClosing(false);
  };

  const closeModal = () => {
    setIsModalOpen(false); // 임시로 그냥 바로 닫히게 설정
  };

  const handleAnimationEnd = () => {
    if (isClosing) {
      setIsModalOpen(false);
      setIsClosing(false);
    }
  };

  useEffect(() => {
    if (project && project !== currentProject) {
      setCurrentProject(project);
    }
  }, [project, currentProject, setCurrentProject]);

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error loading projects</div>;

  return (
    <>
      <S.HomeContentWrapper>
        <S.Title>
          <FolderIcon width={45} height={45} />
          {currentProject?.title || '프로젝트 이름 없음'}
          <S.PythonTitle>python 3.9.6</S.PythonTitle>
          <SettingIcon
            width={22}
            height={22}
            color={Common.colors.gray400}
            onClick={handleSettingClick}
            style={{ cursor: 'pointer' }}
          />
        </S.Title>
        <div>
          <S.SubTitle>
            <WritingIcon width={30} height={30} />
            작업 중인 캔버스
          </S.SubTitle>
          <S.CanvasImage
            src='src/assets/Frame 83.png'
            alt='Canvas Image'
            onClick={handleCanvasClick}
          />
        </div>
        <div>
          <S.SubTitle>
            <ClockIcon width={30} height={30} />
            학습 기록
          </S.SubTitle>
          <S.HistoryWrapper>
            <HistoryList items={historyItems} />
            <Pagination
              currentPage={currentPage}
              totalPages={26}
              onPageChange={setCurrentPage}
            />
          </S.HistoryWrapper>
        </div>
      </S.HomeContentWrapper>
      {isModalOpen && (
        <ProjectModal
          onClose={closeModal}
          isClosing={isClosing}
          onAnimationEnd={handleAnimationEnd}
          type={'modify'}
        />
      )}
    </>
  );
};

export default HomeContent;
