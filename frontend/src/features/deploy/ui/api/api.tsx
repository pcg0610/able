import { useState } from 'react';

import * as S from '@features/deploy/ui/api/api.style'
import Common from '@shared/styles/common';

import InfoContainer from '@features/deploy/ui/common/deploy-info';
import ApiList from '@features/deploy/ui/api/api-list';
import Pagination from '@shared/ui/pagination/pagination';
import { ApiListItem } from '../../type/deploy.type';

const dummyApis: ApiListItem[] = [
  {
    index: 1,
    date: '2024-10-20',
    accuracy: '95.2%',
    status: 'running',
    originDirName: 'project-1'
  },
  {
    index: 2,
    date: '2024-10-21',
    accuracy: '93.4%',
    status: 'stopped',
    originDirName: 'project-2'
  },
  {
    index: 3,
    date: '2024-10-22',
    accuracy: '97.1%',
    status: 'running',
    originDirName: 'project-3'
  },
  {
    index: 4,
    date: '2024-10-23',
    accuracy: '89.5%',
    status: 'failed',
    originDirName: 'project-4'
  }
];

const Api = () => {
  const [currentPage, setCurrentPage] = useState(1);

  return (
    <S.Container>
      <S.TopSection>
        <InfoContainer />

      </S.TopSection>
      <S.List>
        <ApiList apis={dummyApis} />
        <Pagination
          currentPage={1}
          totalPages={20}
          onPageChange={setCurrentPage}
        />
      </S.List>
    </S.Container>
  )
};

export default Api;
