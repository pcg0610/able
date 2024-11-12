import { useNavigate } from 'react-router-dom';

import { ApiListItem } from '@features/deploy/type/deploy.type';
import { useProjectNameStore } from '@entities/project/model/project.model';
import { ApiListWrapper, ApiRow, ApiCell } from '@/features/deploy/ui/api/api-list.style';

import PlayIcon from '@icons/play.svg?react';
import StopIcon from '@icons/stop.svg?react';


interface ApiListProps {
   apis: ApiListItem[];
}

const ApiList = ({ apis }: ApiListProps) => {
   const navigate = useNavigate();
   const { setResultName } = useProjectNameStore();

   const handleApi = (projectName: string, status: string) => {

   };

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
            {apis.map((item, index) => (
               <ApiRow key={item.trainResult} onClick={() => handleApi(item.projectName, item.description)}>
                  <ApiCell width='15%'>{item.uri}</ApiCell>
                  <ApiCell width='30%'>{item.description}</ApiCell>
                  <ApiCell width='25%'>{item.checkpoint}</ApiCell>
                  <ApiCell width='20%'>
                     {item.projectName == "running" ? <PlayIcon width={15} height={15} /> : <StopIcon width={30} height={30} />}
                  </ApiCell>
               </ApiRow>
            ))}
         </tbody>
      </ApiListWrapper>
   );
};

export default ApiList;
