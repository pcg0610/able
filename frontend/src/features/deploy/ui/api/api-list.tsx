import toast from 'react-hot-toast';

import { ApiResponse } from '@features/deploy/type/deploy.type';
import { ApiListWrapper, ApiRow, ApiCell, CellIcon } from '@/features/deploy/ui/api/api-list.style';
import { useStopApi, useRemoveApi } from '@features/deploy/api/use-api.mutation';

import StopIcon from '@icons/stop.svg?react';
import TrashCanIcon from '@icons/trashcan.svg?react';

interface ApiListProps {
   apis: ApiResponse[];
   page: number;
}

const ApiList = ({ apis, page }: ApiListProps) => {
   const { mutate: stopApi } = useStopApi();
   const { mutate: removeApi } = useRemoveApi();

   const handleApi = (uri: string, status: string) => {
      if (status === "running") {
         stopApi({ uri, page }, {
            onSuccess: () => {
               toast.success("API가 중지되었습니다.");
            },
            onError: () => {
               toast.error("에러가 발생했습니다.");
            }
         });
      } else {
         removeApi({ uri, page }, {
            onSuccess: () => {
               toast.success("API가 삭제되었습니다.");
            },
            onError: () => {
               toast.error("에러가 발생했습니다.");
            }
         });
      }
   };

   const rows = apis.length < 5 ? [...apis, ...Array(5 - apis.length).fill(null)] : apis;

   return (
      <ApiListWrapper>
         <thead>
            <tr>
               <ApiCell as="th" width="15%">
                  API경로
               </ApiCell>
               <ApiCell as="th" width="30%">
                  설명
               </ApiCell>
               <ApiCell as="th" width="25%">
                  파라미터
               </ApiCell>
               <ApiCell as="th" width="20%">
                  액션
               </ApiCell>
            </tr>
         </thead>
         <tbody>
            {rows.map((item, index) => (
               item ? (
                  <ApiRow key={item.uri}>
                     <ApiCell width="15%">{item.uri}</ApiCell>
                     <ApiCell width="30%">{item.description}</ApiCell>
                     <ApiCell width="25%">{item.checkpoint}</ApiCell>
                     <ApiCell width="20%" onClick={() => handleApi(item.uri, item.status)}>
                        <CellIcon>
                           {item.status === "running" ? <StopIcon width={30} height={30} /> : <TrashCanIcon width={24} height={24} />}
                        </CellIcon>
                     </ApiCell>
                  </ApiRow>
               ) : (
                  <ApiRow key={`empty-${index}`} style={{ height: '3.125rem' }}>
                     <ApiCell width="15%"></ApiCell>
                     <ApiCell width="30%"></ApiCell>
                     <ApiCell width="25%"></ApiCell>
                     <ApiCell width="20%"></ApiCell>
                  </ApiRow>
               )
            ))}
         </tbody>
      </ApiListWrapper>
   );
};

export default ApiList;
