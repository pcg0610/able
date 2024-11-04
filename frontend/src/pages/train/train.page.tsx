import { useState } from 'react';

import { Container, Content, MainContainer } from '@pages/train/train.style';

import Sidebar from '@/widgets/sidebar/project-sidebar';
import AnalyzeComponent from '@features/train/ui/analyze/analyze';
import ResultComponent from '@features/train/ui/result/result';
import PageHeader from '@widgets/header/page-header';

const TrainPage = () => {
  const [selectedComponent, setSelectedComponent] = useState('result');

  const handleSidebarSelection = (selection: string) => {
    setSelectedComponent(selection);
  };

  return (
    <MainContainer>
      <PageHeader title='프로젝트' date='2024.08.12' />
      <Container>
        <Sidebar onSelectionChange={handleSidebarSelection} type='train' />
        <Content>
          {selectedComponent === 'analyze' && <AnalyzeComponent />}
          {selectedComponent === 'result' && <ResultComponent />}
        </Content>
      </Container>
    </MainContainer>
  );
};

export default TrainPage;
