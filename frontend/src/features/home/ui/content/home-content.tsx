import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

import * as S from '@/features/home/ui/content/home-content.style';
import Common from '@shared/styles/common';
import defaultImage from '@assets/images/default-image.png';
import { useProjectDetail, useProjectHistory } from '@features/home/api/use-home.query';
import { useProjectStore, useProjectNameStore } from '@entities/project/model/project.model';

import HistoryList from '@features/home/ui/content/history-list';
import Pagination from '@shared/ui/pagination/pagination';
import ProjectModal from '@features/home/ui/modal/project-modal';
import WritingIcon from '@icons/writing.svg?react';
import ClockIcon from '@icons/clock.svg?react';
import FolderIcon from '@icons/folder.svg?react';
import SettingIcon from '@icons/setting.svg?react';

const HomeContent = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const [size] = useState(5);

  const navigate = useNavigate();
  const [currentPage, setCurrentPage] = useState(1);

  const { projectName } = useProjectNameStore();
  const { currentProject, setCurrentProject } = useProjectStore();

  const { data: project } = useProjectDetail(projectName);
  const { data: historyData } = useProjectHistory(projectName, currentPage - 1, size);

  const handleCanvasClick = () => {
    navigate('/canvas');
  };

  const handleSettingClick = () => {
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

  useEffect(() => {
    if (project && project !== currentProject) {
      setCurrentProject(project);
    }
  }, [project, currentProject, setCurrentProject]);

  return (
    <>
      <S.HomeContentWrapper>
        <div>
          <S.Title>
            <FolderIcon width={32} height={32} />
            <S.Title>{currentProject?.title || '프로젝트 이름 없음'}</S.Title>
            <SettingIcon
              width={20}
              height={20}
              color={Common.colors.gray400}
              onClick={handleSettingClick}
              style={{ cursor: 'pointer' }}
            />
          </S.Title>
          <S.Description>{currentProject?.description}</S.Description>
        </div>
        <div>
          <S.SubTitle>
            <WritingIcon width={24} height={24} />
            작업 중인 캔버스
          </S.SubTitle>
          <S.CanvasImage
            src={currentProject?.thumbnail || defaultImage}
            alt="Canvas Image"
            onClick={handleCanvasClick}
          />
        </div>
        <div>
          <S.SubTitle>
            <ClockIcon width={24} height={24} />
            학습 기록
          </S.SubTitle>
          {historyData ? (
            <S.HistoryWrapper>
              <HistoryList trainSummaries={historyData?.trainSummaries || []} />
              <Pagination
                currentPage={currentPage}
                totalPages={historyData?.totalTrainLogs || 0}
                onPageChange={setCurrentPage}
              />
            </S.HistoryWrapper>
          ) : (
            <S.Text>학습한 기록이 없어요</S.Text>
          )}
        </div>
      </S.HomeContentWrapper>
      {isModalOpen && (
        <ProjectModal onClose={closeModal} isClosing={isClosing} onAnimationEnd={handleAnimationEnd} type={'modify'} />
      )}
    </>
  );
};

export default HomeContent;
