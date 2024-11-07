import { useState } from 'react';

import { Container, Content, MainContainer } from '@pages/deploy/deploy.style';

import Sidebar from '@/widgets/sidebar/project-sidebar';
import ServerComponent from '@features/deploy/ui/server/server';
import ApiComponent from '@features/deploy/ui/api/api';
import PageHeader from '@widgets/header/page-header';

const TrainPage = () => {
  const [selectedComponent, setSelectedComponent] = useState('result');

  const handleSidebarSelection = (selection: string) => {
    setSelectedComponent(selection);
  };

  return (
    <MainContainer>
      <PageHeader title="프로젝트" date="2024.08.12" />
      <Container>
        <Sidebar onSelectionChange={handleSidebarSelection} type="deploy" />
        <Content>
          {selectedComponent === 'server' && <ServerComponent />}
          {selectedComponent === 'api' && <ApiComponent />}
        </Content>
      </Container>
    </MainContainer>
  );
};

export default TrainPage;
