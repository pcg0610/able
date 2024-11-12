import { useState } from 'react';
import toast from 'react-hot-toast';

import { useProjectNameStore } from '@entities/project/model/project.model';
import type { Option } from '@shared/types/common.type';

import Modal from '@shared/ui/modal/modal';
import Input from '@shared/ui/input/input';
import DropDown from '@shared/ui/dropdown/dropdown';

interface DeviceSelectModalProps {
   onClose: () => void;
   onSubmit: (deviceIndex: number) => void;
}

const DeployModal = ({ onClose, onSubmit }: DeviceSelectModalProps) => {
   const { projectName, resultName, epochName, setEpochName } = useProjectNameStore();
   const [selectedOption, setSelectedOption] = useState<Option | null>(null);

   const options: Option[] = [
      { value: '10.0', label: 'CUDA 10.0', canSelect: true },
      { value: '10.1', label: 'CUDA 10.1', canSelect: true },
      { value: '10.2', label: 'CUDA 10.2', canSelect: false }, // 선택 불가 예시
      { value: '11.0', label: 'CUDA 11.0', canSelect: true },
      { value: '11.1', label: 'CUDA 11.1', canSelect: true },
      { value: '11.2', label: 'CUDA 11.2', canSelect: true },
      { value: '12.0', label: 'CUDA 12.0', canSelect: true }
   ];

   const [device, setDevice] = useState<{ index: number; name: string }>({
      index: -1,
      name: '',
   });

   const handleSelect = (option: Option) => {
      setSelectedOption(option);
   };


   const handleConfigChange = (newDevice: { index: number; name: string }) => {
      setDevice(newDevice);
   };

   const handleSubmit = () => {
      if (device.name) {
         onSubmit(device.index);
         onClose();
         return;
      }
      toast.error('모든 칸을 채워주세요.');
   };

   return (
      <Modal onClose={onClose} onConfirm={handleSubmit} title="배포할 API 정보를 입력하세요" confirmText="확인">
         <DropDown
            label="모델 선택"
            placeholder="배포할 모델을 선택하세요"
            options={options}
            onSelect={handleSelect}
         />
         <Input
            label="API 경로"
            placeholder="ex) /api/v1/model"
            value={''}
            onChange={(e) => handleConfigChange('epoch', 123)}
         />
         <Input
            label="설명"
            placeholder="API에 대한 설명을 입력하세요"
            value={''}
            onChange={(e) => handleConfigChange('epoch', 123)}
         />
      </Modal>
   );
};

export default DeployModal;
