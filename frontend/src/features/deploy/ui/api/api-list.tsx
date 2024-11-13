import { ApiResponse } from '@features/deploy/type/deploy.type';
import { ApiListWrapper, ApiRow, ApiCell } from '@/features/deploy/ui/api/api-list.style';
import { useStopApi } from '@features/deploy/api/use-api.mutation';

import StopIcon from '@icons/stop.svg?react';
import TrashCanIcon from '@icons/trashcan.svg?react';


interface ApiListProps {
   apis: ApiResponse[];
   page: number;
}

const ApiList = ({ apis, page }: ApiListProps) => {
   const { mutate: stopApi } = useStopApi();

   const handleApi = (uri: string, status: string) => {
      if (status === "running") {
         stopApi({ uri, page });
      } else {
         stopApi({ uri, page });
      }
   };

   console.log(apis);

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
         {apis.length > 0 ?
            <tbody>
               {apis.map((item, index) => (
                  <ApiRow key={item.uri}>
                     <ApiCell width='15%'>{item.uri}</ApiCell>
                     <ApiCell width='30%'>{item.description}</ApiCell>
                     <ApiCell width='25%'>{item.checkpoint}</ApiCell>
                     <ApiCell width='20%' onClick={() => handleApi(item.uri, item.status)}>
                        {item.status == "running" ? <StopIcon width={30} height={30} /> : <TrashCanIcon width={24} height={24} />}
                     </ApiCell>
                  </ApiRow>
               ))}
            </tbody>
            : (
               <tbody>
                  <tr>
                     <ApiCell colSpan={4} style={{ textAlign: 'center' }}>
                        데이터 없음
                     </ApiCell>
                  </tr>
               </tbody>
            )}
      </ApiListWrapper>
   );
};

export default ApiList;
